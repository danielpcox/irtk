"""Weight decomposition.

Decomposes weight matrices using SVD and related methods: identifying shared
substructure, weight clustering, spectral analysis, and low-rank patterns.

References:
    Sharma et al. (2023) "The Truth is in There: Improving Reasoning with Layer-Selective Rank Reduction"
    Hsu et al. (2022) "Language Model Decomposition"
"""

import jax
import jax.numpy as jnp
import numpy as np


def weight_svd_decomposition(model, layer, component="attn"):
    """SVD decomposition of weight matrices in a layer.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        component: "attn" or "mlp".

    Returns:
        dict with:
            singular_values: dict of weight_name -> singular values
            effective_ranks: dict of weight_name -> approximate rank
            total_params: int
            compressible_params: int (removable at 1% threshold)
            compression_ratio: float
    """
    svs = {}
    ranks = {}
    total = 0
    compressible = 0

    if component == "attn":
        block = model.blocks[layer].attn
        for name, attr in [("W_Q", "W_Q"), ("W_K", "W_K"), ("W_V", "W_V"), ("W_O", "W_O")]:
            W = np.array(getattr(block, attr))
            # Reshape multi-head to 2D
            orig_shape = W.shape
            W_2d = W.reshape(-1, W.shape[-1])
            U, s, Vt = np.linalg.svd(W_2d, full_matrices=False)
            svs[name] = s
            s_norm = s / (s[0] + 1e-10)
            ranks[name] = float(np.sum(s_norm > 0.01))
            total += W_2d.shape[0] * W_2d.shape[1]
            # Compressible: params in singular values below 1% of max
            n_keep = int(np.sum(s_norm > 0.01))
            compressible += W_2d.shape[0] * W_2d.shape[1] - n_keep * (W_2d.shape[0] + W_2d.shape[1])
    else:
        block = model.blocks[layer].mlp
        for name, attr in [("W_in", "W_in"), ("W_out", "W_out")]:
            W = np.array(getattr(block, attr))
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            svs[name] = s
            s_norm = s / (s[0] + 1e-10)
            ranks[name] = float(np.sum(s_norm > 0.01))
            total += W.shape[0] * W.shape[1]
            n_keep = int(np.sum(s_norm > 0.01))
            compressible += W.shape[0] * W.shape[1] - n_keep * (W.shape[0] + W.shape[1])

    compressible = max(0, compressible)
    ratio = compressible / (total + 1e-10)

    return {
        "singular_values": svs,
        "effective_ranks": ranks,
        "total_params": total,
        "compressible_params": compressible,
        "compression_ratio": float(ratio),
    }


def weight_shared_substructure(model):
    """Find shared substructure across layers in weight matrices.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            cross_layer_similarity: [n_layers, n_layers] cosine similarity of flattened weights
            most_similar_layers: tuple (layer_a, layer_b)
            shared_subspace_dim: int, dimension of shared subspace
            layer_weight_norms: [n_layers] norm of all weights per layer
    """
    n_layers = model.cfg.n_layers

    # Flatten weights per layer
    layer_vecs = []
    norms = np.zeros(n_layers)

    for layer in range(n_layers):
        block = model.blocks[layer]
        parts = []
        for W in [block.attn.W_Q, block.attn.W_K, block.attn.W_V, block.attn.W_O,
                   block.mlp.W_in, block.mlp.W_out]:
            parts.append(np.array(W).flatten())
        vec = np.concatenate(parts)
        norms[layer] = float(np.linalg.norm(vec))
        layer_vecs.append(vec)

    # Cross-layer similarity
    sim = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            ni = np.linalg.norm(layer_vecs[i]) + 1e-10
            nj = np.linalg.norm(layer_vecs[j]) + 1e-10
            sim[i, j] = float(np.dot(layer_vecs[i], layer_vecs[j]) / (ni * nj))

    # Most similar pair (off-diagonal)
    mask = np.ones_like(sim) - np.eye(n_layers)
    sim_masked = sim * mask - (1 - mask) * 10
    best = np.unravel_index(np.argmax(sim_masked), sim.shape)

    # Shared subspace dimension via PCA
    stacked = np.array(layer_vecs)
    if n_layers > 1:
        centered = stacked - np.mean(stacked, axis=0)
        cov = centered @ centered.T
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals_norm = evals / (evals[0] + 1e-10)
        shared_dim = int(np.sum(evals_norm > 0.1))
    else:
        shared_dim = 1

    return {
        "cross_layer_similarity": sim,
        "most_similar_layers": (int(best[0]), int(best[1])),
        "shared_subspace_dim": shared_dim,
        "layer_weight_norms": norms,
    }


def weight_clustering(model, n_clusters=3):
    """Cluster weight matrices by similarity.

    Args:
        model: HookedTransformer model.
        n_clusters: Number of clusters.

    Returns:
        dict with:
            labels: dict of (layer, component) -> cluster label
            cluster_sizes: list of cluster sizes
            cluster_centers: [n_clusters, n_features] (in PCA space)
            within_cluster_variance: [n_clusters]
    """
    n_layers = model.cfg.n_layers

    # Collect weight vectors
    weight_vecs = []
    weight_names = []

    for layer in range(n_layers):
        block = model.blocks[layer]
        for name, W in [("attn", np.concatenate([np.array(block.attn.W_Q).flatten(),
                                                  np.array(block.attn.W_K).flatten(),
                                                  np.array(block.attn.W_V).flatten(),
                                                  np.array(block.attn.W_O).flatten()])),
                         ("mlp", np.concatenate([np.array(block.mlp.W_in).flatten(),
                                                 np.array(block.mlp.W_out).flatten()]))]:
            weight_vecs.append(W)
            weight_names.append((layer, name))

    n_items = len(weight_vecs)
    n_clusters = min(n_clusters, n_items)

    # Pad all vectors to same length for clustering
    max_len = max(len(v) for v in weight_vecs)
    padded = []
    for v in weight_vecs:
        if len(v) < max_len:
            padded.append(np.concatenate([v, np.zeros(max_len - len(v))]))
        else:
            padded.append(v)

    # Normalize vectors
    normed = []
    for v in padded:
        n = np.linalg.norm(v) + 1e-10
        normed.append(v / n)

    # Initialize centers
    np.random.seed(42)
    center_idx = np.random.choice(n_items, size=n_clusters, replace=False)
    centers = [normed[i].copy() for i in center_idx]

    # Iterate
    for _ in range(10):
        # Assign
        labels_arr = np.zeros(n_items, dtype=int)
        for i in range(n_items):
            sims = [float(np.dot(normed[i], c)) for c in centers]
            labels_arr[i] = int(np.argmax(sims))

        # Update centers
        for c in range(n_clusters):
            members = [normed[i] for i in range(n_items) if labels_arr[i] == c]
            if members:
                centers[c] = np.mean(members, axis=0)
                centers[c] /= np.linalg.norm(centers[c]) + 1e-10

    labels = {weight_names[i]: int(labels_arr[i]) for i in range(n_items)}
    sizes = [int(np.sum(labels_arr == c)) for c in range(n_clusters)]

    # Within-cluster variance
    variances = np.zeros(n_clusters)
    for c in range(n_clusters):
        members = [normed[i] for i in range(n_items) if labels_arr[i] == c]
        if len(members) > 1:
            dists = [float(1 - np.dot(m, centers[c])) for m in members]
            variances[c] = float(np.mean(dists))

    return {
        "labels": labels,
        "cluster_sizes": sizes,
        "cluster_centers": np.array(centers)[:, :10],  # truncate for storage
        "within_cluster_variance": variances,
    }


def spectral_weight_analysis(model):
    """Analyze spectral properties of weight matrices across the model.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            condition_numbers: dict of (layer, component) -> condition number
            spectral_norms: dict of (layer, component) -> spectral norm
            rank_deficiency: dict of (layer, component) -> fraction of near-zero SVs
            worst_conditioned: tuple (layer, component)
    """
    n_layers = model.cfg.n_layers

    cond_nums = {}
    spec_norms = {}
    rank_def = {}
    worst_cond = 0.0
    worst_name = (0, "attn")

    for layer in range(n_layers):
        block = model.blocks[layer]

        for name, W_list in [("attn_Q", [block.attn.W_Q]),
                              ("attn_K", [block.attn.W_K]),
                              ("attn_V", [block.attn.W_V]),
                              ("mlp_in", [block.mlp.W_in]),
                              ("mlp_out", [block.mlp.W_out])]:
            W = np.array(W_list[0])
            W_2d = W.reshape(-1, W.shape[-1])
            s = np.linalg.svd(W_2d, compute_uv=False)

            cn = float(s[0] / (s[-1] + 1e-10))
            cond_nums[(layer, name)] = cn
            spec_norms[(layer, name)] = float(s[0])

            s_norm = s / (s[0] + 1e-10)
            rank_def[(layer, name)] = float(np.mean(s_norm < 0.01))

            if cn > worst_cond:
                worst_cond = cn
                worst_name = (layer, name)

    return {
        "condition_numbers": cond_nums,
        "spectral_norms": spec_norms,
        "rank_deficiency": rank_def,
        "worst_conditioned": worst_name,
    }


def weight_norm_distribution(model):
    """Analyze the distribution of weight norms across the model.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            component_norms: dict of (layer, component) -> float
            attn_norms_by_layer: [n_layers]
            mlp_norms_by_layer: [n_layers]
            total_norm: float
            norm_ratio_attn_mlp: [n_layers] ratio of attn/mlp norms
    """
    n_layers = model.cfg.n_layers

    comp_norms = {}
    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        block = model.blocks[layer]

        attn_parts = []
        for W in [block.attn.W_Q, block.attn.W_K, block.attn.W_V, block.attn.W_O]:
            attn_parts.append(np.array(W).flatten())
        attn_vec = np.concatenate(attn_parts)
        an = float(np.linalg.norm(attn_vec))
        comp_norms[(layer, "attn")] = an
        attn_norms[layer] = an

        mlp_parts = [np.array(block.mlp.W_in).flatten(), np.array(block.mlp.W_out).flatten()]
        mlp_vec = np.concatenate(mlp_parts)
        mn = float(np.linalg.norm(mlp_vec))
        comp_norms[(layer, "mlp")] = mn
        mlp_norms[layer] = mn

    total = float(np.sqrt(np.sum(attn_norms**2 + mlp_norms**2)))
    ratio = attn_norms / (mlp_norms + 1e-10)

    return {
        "component_norms": comp_norms,
        "attn_norms_by_layer": attn_norms,
        "mlp_norms_by_layer": mlp_norms,
        "total_norm": total,
        "norm_ratio_attn_mlp": ratio,
    }
