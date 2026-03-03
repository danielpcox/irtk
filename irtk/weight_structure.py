"""Weight matrix structure analysis.

Analyzes spectral properties of weight matrices, parameter utilization
efficiency, weight sharing patterns, and structural properties.

References:
    Martin & Mahoney (2021) "Implicit Self-Regularization in Deep Neural Networks"
    Sharma & Kaplan (2022) "Scaling Laws from the Data Manifold Dimension"
"""

import jax
import jax.numpy as jnp
import numpy as np


def weight_spectral_analysis(model, top_k=5):
    """Analyze spectral properties of key weight matrices.

    Computes singular value distributions for attention and MLP weights.

    Args:
        model: HookedTransformer model.
        top_k: Number of top singular values to return.

    Returns:
        dict with:
            attn_spectral: dict (layer, matrix_name) -> top_k singular values
            mlp_spectral: dict (layer, matrix_name) -> top_k singular values
            effective_ranks: dict of matrix name -> effective rank
            condition_numbers: dict of matrix name -> condition number
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_spectral = {}
    mlp_spectral = {}
    eff_ranks = {}
    cond_numbers = {}

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp

        # Attention matrices: stack heads
        for name, W in [("W_Q", attn.W_Q), ("W_K", attn.W_K), ("W_V", attn.W_V), ("W_O", attn.W_O)]:
            mat = np.array(W).reshape(-1, W.shape[-1])  # flatten heads
            S = np.linalg.svd(mat, compute_uv=False)
            key = f"L{layer}_{name}"
            attn_spectral[(layer, name)] = S[:top_k].tolist()

            # Effective rank
            S2 = S ** 2
            total = np.sum(S2)
            if total > 1e-10:
                probs = S2 / total
                probs = probs[probs > 1e-12]
                eff_ranks[key] = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
            else:
                eff_ranks[key] = 0.0

            # Condition number
            if S[-1] > 1e-10:
                cond_numbers[key] = float(S[0] / S[-1])
            else:
                cond_numbers[key] = float('inf')

        # MLP matrices
        for name, W in [("W_in", mlp.W_in), ("W_out", mlp.W_out)]:
            mat = np.array(W)
            S = np.linalg.svd(mat, compute_uv=False)
            key = f"L{layer}_{name}"
            mlp_spectral[(layer, name)] = S[:top_k].tolist()

            S2 = S ** 2
            total = np.sum(S2)
            if total > 1e-10:
                probs = S2 / total
                probs = probs[probs > 1e-12]
                eff_ranks[key] = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
            else:
                eff_ranks[key] = 0.0

            if S[-1] > 1e-10:
                cond_numbers[key] = float(S[0] / S[-1])
            else:
                cond_numbers[key] = float('inf')

    return {
        "attn_spectral": attn_spectral,
        "mlp_spectral": mlp_spectral,
        "effective_ranks": eff_ranks,
        "condition_numbers": cond_numbers,
    }


def parameter_utilization(model):
    """Measure parameter utilization efficiency.

    Identifies parameters that are near-zero (potentially prunable)
    and measures overall weight distribution statistics.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            total_params: int
            near_zero_fraction: float, fraction of params with |w| < 0.01
            weight_norm_by_layer: array [n_layers] of total weight norm per layer
            param_count_by_type: dict mapping type -> count
            weight_magnitude_stats: dict with mean, std, min, max
    """
    n_layers = model.cfg.n_layers

    all_params = []
    type_counts = {"attn": 0, "mlp": 0, "embed": 0, "unembed": 0, "ln": 0}
    layer_norms = np.zeros(n_layers)

    # Embedding
    embed_params = np.array(model.embed.W_E).flatten()
    all_params.append(embed_params)
    type_counts["embed"] += len(embed_params)

    # Positional embedding
    pos_params = np.array(model.pos_embed.W_pos).flatten()
    all_params.append(pos_params)
    type_counts["embed"] += len(pos_params)

    # Unembed
    unembed_params = np.array(model.unembed.W_U).flatten()
    all_params.append(unembed_params)
    type_counts["unembed"] += len(unembed_params)

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp

        attn_total = 0
        for W in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
            flat = np.array(W).flatten()
            all_params.append(flat)
            attn_total += len(flat)
            layer_norms[layer] += float(np.sum(flat ** 2))

        type_counts["attn"] += attn_total

        mlp_total = 0
        for W in [mlp.W_in, mlp.W_out]:
            flat = np.array(W).flatten()
            all_params.append(flat)
            mlp_total += len(flat)
            layer_norms[layer] += float(np.sum(flat ** 2))

        type_counts["mlp"] += mlp_total

    layer_norms = np.sqrt(layer_norms)
    all_flat = np.concatenate(all_params)

    return {
        "total_params": len(all_flat),
        "near_zero_fraction": float(np.mean(np.abs(all_flat) < 0.01)),
        "weight_norm_by_layer": layer_norms,
        "param_count_by_type": type_counts,
        "weight_magnitude_stats": {
            "mean": float(np.mean(np.abs(all_flat))),
            "std": float(np.std(all_flat)),
            "min": float(np.min(np.abs(all_flat))),
            "max": float(np.max(np.abs(all_flat))),
        },
    }


def head_weight_similarity(model):
    """Compare weight matrices across attention heads.

    Measures how similar the QKV weight matrices are across heads within
    and across layers.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            within_layer_similarity: array [n_layers] of mean head-pair cosine similarity
            across_layer_similarity: float, mean similarity across all head pairs
            most_similar_heads: tuple ((l1,h1), (l2,h2)) with highest similarity
            most_different_heads: tuple with lowest similarity
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Concatenate QKV weights for each head as its "signature"
    head_vecs = []
    head_labels = []

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        for h in range(n_heads):
            q = np.array(attn.W_Q[h]).flatten()
            k = np.array(attn.W_K[h]).flatten()
            v = np.array(attn.W_V[h]).flatten()
            vec = np.concatenate([q, k, v])
            norm = np.linalg.norm(vec) + 1e-10
            head_vecs.append(vec / norm)
            head_labels.append((layer, h))

    n = len(head_vecs)
    if n < 2:
        return {
            "within_layer_similarity": np.zeros(n_layers),
            "across_layer_similarity": 0.0,
            "most_similar_heads": ((0, 0), (0, 1)),
            "most_different_heads": ((0, 0), (0, 1)),
        }

    mat = np.stack(head_vecs)
    sim = mat @ mat.T

    # Within-layer
    within = np.zeros(n_layers)
    for layer in range(n_layers):
        idxs = [i for i, (l, h) in enumerate(head_labels) if l == layer]
        sims = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                sims.append(sim[idxs[i], idxs[j]])
        within[layer] = np.mean(sims) if sims else 0.0

    # All pairs
    best_sim = -2.0
    worst_sim = 2.0
    best_pair = ((0, 0), (0, 1))
    worst_pair = ((0, 0), (0, 1))
    all_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j]
            all_sims.append(s)
            if s > best_sim:
                best_sim = s
                best_pair = (head_labels[i], head_labels[j])
            if s < worst_sim:
                worst_sim = s
                worst_pair = (head_labels[i], head_labels[j])

    return {
        "within_layer_similarity": within,
        "across_layer_similarity": float(np.mean(all_sims)) if all_sims else 0.0,
        "most_similar_heads": best_pair,
        "most_different_heads": worst_pair,
    }


def embedding_weight_relationship(model):
    """Analyze the relationship between embedding and unembedding weights.

    Tests whether W_U ≈ W_E^T (weight tying) and measures the degree of alignment.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            weight_tying_score: float, mean cosine similarity between W_E rows and W_U columns
            frobenius_distance: float, ||W_E^T - W_U||_F normalized
            rank_correlation: float, Spearman correlation of embed vs unembed norms
            embed_rank: float, effective rank of W_E
            unembed_rank: float, effective rank of W_U
    """
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    d_vocab = W_E.shape[0]

    # Weight tying: cosine of each token's embed row vs unembed column
    E_norms = np.linalg.norm(W_E, axis=1) + 1e-10
    U_norms = np.linalg.norm(W_U, axis=0) + 1e-10
    cosines = np.sum(W_E * W_U.T, axis=1) / (E_norms * U_norms)
    tying_score = float(np.mean(cosines))

    # Frobenius distance
    diff = W_E.T - W_U
    frob = float(np.linalg.norm(diff))
    norm_factor = float(np.linalg.norm(W_E.T) + np.linalg.norm(W_U)) / 2 + 1e-10
    frob_normalized = frob / norm_factor

    # Rank correlation of norms
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(E_norms, U_norms)
        rank_corr = float(corr)
    except Exception:
        rank_corr = float(np.corrcoef(E_norms, U_norms)[0, 1])

    # Effective ranks
    def eff_rank(mat):
        S = np.linalg.svd(mat, compute_uv=False)
        S2 = S ** 2
        total = np.sum(S2)
        if total < 1e-10:
            return 0.0
        probs = S2 / total
        probs = probs[probs > 1e-12]
        return float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    return {
        "weight_tying_score": tying_score,
        "frobenius_distance": frob_normalized,
        "rank_correlation": rank_corr,
        "embed_rank": eff_rank(W_E),
        "unembed_rank": eff_rank(W_U),
    }


def layer_weight_norm_profile(model):
    """Profile weight norms across layers and component types.

    Args:
        model: HookedTransformer model.

    Returns:
        dict with:
            attn_q_norms: array [n_layers] of W_Q frobenius norms
            attn_k_norms: array [n_layers] of W_K frobenius norms
            attn_v_norms: array [n_layers] of W_V frobenius norms
            attn_o_norms: array [n_layers] of W_O frobenius norms
            mlp_in_norms: array [n_layers] of W_in frobenius norms
            mlp_out_norms: array [n_layers] of W_out frobenius norms
            total_per_layer: array [n_layers] of total weight norm per layer
    """
    n_layers = model.cfg.n_layers

    q_norms = np.zeros(n_layers)
    k_norms = np.zeros(n_layers)
    v_norms = np.zeros(n_layers)
    o_norms = np.zeros(n_layers)
    win_norms = np.zeros(n_layers)
    wout_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp
        q_norms[layer] = float(np.linalg.norm(np.array(attn.W_Q)))
        k_norms[layer] = float(np.linalg.norm(np.array(attn.W_K)))
        v_norms[layer] = float(np.linalg.norm(np.array(attn.W_V)))
        o_norms[layer] = float(np.linalg.norm(np.array(attn.W_O)))
        win_norms[layer] = float(np.linalg.norm(np.array(mlp.W_in)))
        wout_norms[layer] = float(np.linalg.norm(np.array(mlp.W_out)))

    total = q_norms + k_norms + v_norms + o_norms + win_norms + wout_norms

    return {
        "attn_q_norms": q_norms,
        "attn_k_norms": k_norms,
        "attn_v_norms": v_norms,
        "attn_o_norms": o_norms,
        "mlp_in_norms": win_norms,
        "mlp_out_norms": wout_norms,
        "total_per_layer": total,
    }
