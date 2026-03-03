"""Geometric properties of activation spaces.

Analyzes curvature, manifold dimensionality, representation similarity
across inputs, and geometric structure of hidden state spaces.

References:
    Ansuini et al. (2019) "Intrinsic Dimension of Data Representations in Deep Neural Networks"
    Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
"""

import jax
import jax.numpy as jnp
import numpy as np


def activation_manifold_dimension(model, tokens_list, layer=-1, pos=-1):
    """Estimate the intrinsic dimensionality of activations at a layer.

    Uses PCA-based estimation on activations from multiple inputs.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays to compute activations for.
        layer: Layer to analyze (-1 = final).
        pos: Position.

    Returns:
        dict with:
            intrinsic_dim: float, estimated intrinsic dimensionality
            explained_variance: array of variance ratios
            cumulative_variance: array of cumulative variance
            n_for_90pct: int, dimensions needed for 90% variance
            n_for_99pct: int, dimensions needed for 99% variance
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if layer == -1:
        layer = n_layers - 1
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tokens in tokens_list:
        state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=state)
        act = state.cache.get(hook_name)
        if act is not None:
            activations.append(np.array(act[pos]))

    if len(activations) < 2:
        d = model.cfg.d_model
        return {
            "intrinsic_dim": float(d),
            "explained_variance": np.ones(1),
            "cumulative_variance": np.ones(1),
            "n_for_90pct": d,
            "n_for_99pct": d,
        }

    matrix = np.stack(activations)  # [n_samples, d_model]
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)

    S = np.linalg.svd(centered, compute_uv=False)
    S2 = S ** 2
    total = np.sum(S2)

    if total < 1e-10:
        return {
            "intrinsic_dim": 0.0,
            "explained_variance": np.zeros(len(S)),
            "cumulative_variance": np.zeros(len(S)),
            "n_for_90pct": 0,
            "n_for_99pct": 0,
        }

    var_ratios = S2 / total
    cumulative = np.cumsum(var_ratios)

    # Effective rank as intrinsic dimension
    probs = var_ratios[var_ratios > 1e-12]
    intrinsic = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    n_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    n_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    return {
        "intrinsic_dim": intrinsic,
        "explained_variance": var_ratios,
        "cumulative_variance": cumulative,
        "n_for_90pct": min(n_90, len(S)),
        "n_for_99pct": min(n_99, len(S)),
    }


def representation_similarity_across_inputs(model, tokens_a, tokens_b, pos=-1):
    """Compare representations of two inputs across layers.

    Measures cosine similarity of residual stream states for two different
    inputs at each layer. Shows where representations diverge or converge.

    Args:
        model: HookedTransformer model.
        tokens_a: First input [seq_len].
        tokens_b: Second input [seq_len].
        pos: Position to compare.

    Returns:
        dict with:
            layer_similarities: array [n_layers+1] of cosine similarities
            divergence_layer: int, layer where similarity drops most
            convergence_layer: int, layer where similarity increases most
            initial_similarity: float
            final_similarity: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    state_a = HookState(hook_fns={}, cache={})
    model(tokens_a, hook_state=state_a)

    state_b = HookState(hook_fns={}, cache={})
    model(tokens_b, hook_state=state_b)

    similarities = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"

        act_a = state_a.cache.get(key)
        act_b = state_b.cache.get(key)

        if act_a is not None and act_b is not None:
            va = np.array(act_a[pos])
            vb = np.array(act_b[pos])
            na = np.linalg.norm(va) + 1e-10
            nb = np.linalg.norm(vb) + 1e-10
            similarities[layer] = float(np.dot(va, vb) / (na * nb))

    # Changes
    diffs = np.diff(similarities)
    divergence = int(np.argmin(diffs)) if len(diffs) > 0 else 0
    convergence = int(np.argmax(diffs)) if len(diffs) > 0 else 0

    return {
        "layer_similarities": similarities,
        "divergence_layer": divergence,
        "convergence_layer": convergence,
        "initial_similarity": float(similarities[0]),
        "final_similarity": float(similarities[-1]),
    }


def activation_cluster_analysis(model, tokens_list, layer=-1, pos=-1, n_clusters=3):
    """Cluster activations from multiple inputs.

    Uses k-means to identify groups of inputs with similar internal
    representations.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Layer to analyze.
        pos: Position.
        n_clusters: Number of clusters.

    Returns:
        dict with:
            cluster_assignments: array of cluster labels per input
            cluster_sizes: array of cluster sizes
            within_cluster_similarity: array of mean pairwise similarity per cluster
            between_cluster_similarity: float
            cluster_centroids: array [n_clusters, d_model]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if layer == -1:
        layer = n_layers - 1
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tokens in tokens_list:
        state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=state)
        act = state.cache.get(hook_name)
        if act is not None:
            activations.append(np.array(act[pos]))

    if len(activations) < n_clusters:
        n_clusters = max(1, len(activations))

    matrix = np.stack(activations)
    n = matrix.shape[0]

    # Simple k-means
    rng = np.random.RandomState(42)
    indices = rng.choice(n, size=min(n_clusters, n), replace=False)
    centroids = matrix[indices].copy()

    for _ in range(20):
        # Assign
        dists = np.zeros((n, n_clusters))
        for k in range(n_clusters):
            dists[:, k] = np.linalg.norm(matrix - centroids[k], axis=1)
        assignments = np.argmin(dists, axis=1)

        # Update
        for k in range(n_clusters):
            mask = assignments == k
            if np.any(mask):
                centroids[k] = np.mean(matrix[mask], axis=0)

    # Cluster statistics
    sizes = np.array([np.sum(assignments == k) for k in range(n_clusters)])

    # Normalize for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms

    within_sim = np.zeros(n_clusters)
    for k in range(n_clusters):
        mask = assignments == k
        if np.sum(mask) >= 2:
            cluster_vecs = normed[mask]
            sim = cluster_vecs @ cluster_vecs.T
            n_k = cluster_vecs.shape[0]
            mask_upper = np.triu(np.ones((n_k, n_k), dtype=bool), k=1)
            within_sim[k] = np.mean(sim[mask_upper])

    # Between-cluster similarity
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
    normed_centroids = centroids / centroid_norms
    c_sim = normed_centroids @ normed_centroids.T
    mask_upper = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
    between_sim = float(np.mean(c_sim[mask_upper])) if np.any(mask_upper) else 0.0

    return {
        "cluster_assignments": assignments,
        "cluster_sizes": sizes,
        "within_cluster_similarity": within_sim,
        "between_cluster_similarity": between_sim,
        "cluster_centroids": centroids,
    }


def representation_curvature(model, tokens, pos=-1):
    """Estimate curvature of the representation trajectory through layers.

    High curvature means the representation is changing direction rapidly
    between layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            curvatures: array [n_layers-1] of curvature at each point
            max_curvature_layer: int
            mean_curvature: float
            trajectory_length: float, total path length through representation space
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    states = []
    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            states.append(np.array(resid[pos]))
        else:
            states.append(np.zeros(model.cfg.d_model))

    # Compute velocity vectors (differences)
    velocities = []
    for i in range(len(states) - 1):
        velocities.append(states[i + 1] - states[i])

    # Curvature = angle change between consecutive velocity vectors
    curvatures = np.zeros(max(0, len(velocities) - 1))
    for i in range(len(curvatures)):
        v1 = velocities[i]
        v2 = velocities[i + 1]
        n1 = np.linalg.norm(v1) + 1e-10
        n2 = np.linalg.norm(v2) + 1e-10
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        curvatures[i] = float(np.arccos(cos_angle))

    # Total path length
    lengths = [np.linalg.norm(v) for v in velocities]
    total_length = sum(lengths)

    max_curv = int(np.argmax(curvatures)) if len(curvatures) > 0 else 0

    return {
        "curvatures": curvatures,
        "max_curvature_layer": max_curv,
        "mean_curvature": float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0,
        "trajectory_length": float(total_length),
    }


def activation_norm_distribution(model, tokens_list, layer=-1, pos=-1):
    """Analyze the distribution of activation norms across inputs.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Layer to analyze.
        pos: Position.

    Returns:
        dict with:
            norms: array of activation norms per input
            mean_norm: float
            std_norm: float
            min_norm: float
            max_norm: float
            coefficient_of_variation: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if layer == -1:
        layer = n_layers - 1
    hook_name = f"blocks.{layer}.hook_resid_post"

    norms = []
    for tokens in tokens_list:
        state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=state)
        act = state.cache.get(hook_name)
        if act is not None:
            norms.append(float(np.linalg.norm(np.array(act[pos]))))

    norms = np.array(norms)
    mean = float(np.mean(norms)) if len(norms) > 0 else 0.0
    std = float(np.std(norms)) if len(norms) > 0 else 0.0

    return {
        "norms": norms,
        "mean_norm": mean,
        "std_norm": std,
        "min_norm": float(np.min(norms)) if len(norms) > 0 else 0.0,
        "max_norm": float(np.max(norms)) if len(norms) > 0 else 0.0,
        "coefficient_of_variation": std / (mean + 1e-10),
    }
