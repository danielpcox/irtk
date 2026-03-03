"""Feature interaction maps for mechanistic interpretability.

Tools for understanding how features interact within the model:
coactivation analysis, mutual suppression/enhancement detection,
feature clustering, interaction strength measurement, and
dependency graph construction.

References:
- Elhage et al. (2022) "Toy Models of Superposition"
- Bricken et al. (2023) "Towards Monosemanticity"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def feature_coactivation(
    model,
    tokens_list: list,
    layer: int = -1,
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Analyze which residual stream dimensions coactivate.

    Computes the coactivation matrix across inputs to find dimensions
    that tend to be active together.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Which layer's residual output to analyze (-1 for last).
        pos: Which position to analyze (-1 for last).
        top_k: Number of top coactivating pairs to return.

    Returns:
        Dict with coactivation_matrix, top_pairs, activation_correlation.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tok in tokens_list:
        cache = {}
        hook_state = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hook_state)
        act = np.array(cache[hook_name][pos])  # [d_model]
        activations.append(act)
    activations = np.stack(activations)  # [n_inputs, d_model]

    # Correlation matrix
    means = activations.mean(axis=0, keepdims=True)
    centered = activations - means
    stds = np.std(activations, axis=0, keepdims=True) + 1e-10
    normed = centered / stds
    corr = (normed.T @ normed) / len(activations)

    # Find top coactivating pairs (off-diagonal)
    d = corr.shape[0]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((i, j, float(corr[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = [(i, j, c) for i, j, c in pairs[:top_k]]

    # Binary coactivation (both above mean)
    binary = (activations > means).astype(np.float32)
    coact = (binary.T @ binary) / len(activations)

    return {
        "coactivation_matrix": jnp.array(coact),
        "correlation_matrix": jnp.array(corr),
        "top_pairs": top_pairs,
        "mean_activations": jnp.array(means.squeeze()),
        "n_inputs": len(tokens_list),
    }


def mutual_suppression_enhancement(
    model,
    tokens_list: list,
    metric_fn: Callable,
    layer: int = -1,
    n_directions: int = 5,
) -> dict:
    """Detect mutual suppression and enhancement between directions.

    Tests pairs of top-variance directions to see if they suppress or
    enhance each other's effect on the metric.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        metric_fn: fn(logits, tokens) -> scalar.
        layer: Layer to intervene on.
        n_directions: Number of top-variance directions to test.

    Returns:
        Dict with interaction_matrix, suppressive_pairs, enhancing_pairs.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Collect activations and find top-variance directions
    activations = []
    for tok in tokens_list:
        cache = {}
        hook_state = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hook_state)
        activations.append(np.array(cache[hook_name][-1]))
    act_matrix = np.stack(activations)  # [n, d_model]

    # PCA for top directions
    centered = act_matrix - act_matrix.mean(axis=0)
    cov = centered.T @ centered / len(act_matrix)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_dirs = eigvecs[:, -n_directions:][:, ::-1]  # [d_model, n_dirs]

    # Baseline scores
    baselines = []
    for tok in tokens_list:
        logits = model(tok)
        baselines.append(float(metric_fn(logits, tok)))
    baselines = np.array(baselines)
    mean_baseline = float(np.mean(baselines))

    # Single-direction ablation effects
    single_effects = np.zeros(n_directions)
    for d_idx in range(n_directions):
        direction = jnp.array(top_dirs[:, d_idx])

        def remove_dir(x, name, _dir=direction):
            proj = (x[-1] @ _dir) * _dir
            return x.at[-1].set(x[-1] - proj)

        effects = []
        for tok in tokens_list:
            hook_state = HookState(hook_fns={hook_name: remove_dir}, cache=None)
            logits = model(tok, hook_state=hook_state)
            effects.append(float(metric_fn(logits, tok)))
        single_effects[d_idx] = mean_baseline - float(np.mean(effects))

    # Pairwise ablation effects
    interaction_matrix = np.zeros((n_directions, n_directions))
    for i in range(n_directions):
        interaction_matrix[i, i] = single_effects[i]
        for j in range(i + 1, n_directions):
            dir_i = jnp.array(top_dirs[:, i])
            dir_j = jnp.array(top_dirs[:, j])

            def remove_both(x, name, _di=dir_i, _dj=dir_j):
                proj_i = (x[-1] @ _di) * _di
                proj_j = (x[-1] @ _dj) * _dj
                return x.at[-1].set(x[-1] - proj_i - proj_j)

            effects = []
            for tok in tokens_list:
                hook_state = HookState(hook_fns={hook_name: remove_both}, cache=None)
                logits = model(tok, hook_state=hook_state)
                effects.append(float(metric_fn(logits, tok)))
            joint_effect = mean_baseline - float(np.mean(effects))

            # Interaction = joint - (sum of singles)
            interaction = joint_effect - (single_effects[i] + single_effects[j])
            interaction_matrix[i, j] = interaction
            interaction_matrix[j, i] = interaction

    # Classify pairs
    suppressive = []
    enhancing = []
    for i in range(n_directions):
        for j in range(i + 1, n_directions):
            val = interaction_matrix[i, j]
            if val < -1e-6:
                suppressive.append((i, j, float(val)))
            elif val > 1e-6:
                enhancing.append((i, j, float(val)))

    return {
        "interaction_matrix": jnp.array(interaction_matrix),
        "single_effects": jnp.array(single_effects),
        "suppressive_pairs": sorted(suppressive, key=lambda x: x[2]),
        "enhancing_pairs": sorted(enhancing, key=lambda x: -x[2]),
        "n_directions": n_directions,
    }


def feature_clustering(
    model,
    tokens_list: list,
    layer: int = -1,
    pos: int = -1,
    n_clusters: int = 4,
) -> dict:
    """Cluster residual stream dimensions by coactivation patterns.

    Uses simple k-means-style clustering on the correlation structure
    of activation dimensions.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Layer to analyze.
        pos: Position to analyze.
        n_clusters: Number of clusters.

    Returns:
        Dict with cluster_assignments, cluster_sizes, within_cluster_correlation,
        between_cluster_correlation.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tok in tokens_list:
        cache = {}
        hook_state = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hook_state)
        activations.append(np.array(cache[hook_name][pos]))
    act_matrix = np.stack(activations)  # [n_inputs, d_model]

    # Correlation matrix of dimensions
    stds = np.std(act_matrix, axis=0, keepdims=True) + 1e-10
    normed = (act_matrix - act_matrix.mean(axis=0, keepdims=True)) / stds
    corr = normed.T @ normed / len(act_matrix)  # [d_model, d_model]

    # Simple k-means on correlation profiles
    d = corr.shape[0]
    n_clusters = min(n_clusters, d)
    rng = np.random.RandomState(42)
    assignments = rng.randint(0, n_clusters, size=d)

    for _ in range(20):
        # Compute centroids
        centroids = np.zeros((n_clusters, d))
        for c in range(n_clusters):
            mask = assignments == c
            if mask.any():
                centroids[c] = corr[mask].mean(axis=0)
        # Reassign
        new_assignments = np.zeros(d, dtype=int)
        for i in range(d):
            dists = np.array([np.sum((corr[i] - centroids[c]) ** 2) for c in range(n_clusters)])
            new_assignments[i] = int(np.argmin(dists))
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

    # Cluster stats
    cluster_sizes = [int(np.sum(assignments == c)) for c in range(n_clusters)]

    within_corr = []
    between_corr = []
    for c in range(n_clusters):
        mask = assignments == c
        if np.sum(mask) > 1:
            sub = corr[np.ix_(mask, mask)]
            # Upper triangle mean
            triu = sub[np.triu_indices(sub.shape[0], k=1)]
            within_corr.append(float(np.mean(triu)) if len(triu) > 0 else 0.0)
        else:
            within_corr.append(0.0)

    for ci in range(n_clusters):
        for cj in range(ci + 1, n_clusters):
            mask_i = assignments == ci
            mask_j = assignments == cj
            sub = corr[np.ix_(mask_i, mask_j)]
            between_corr.append(float(np.mean(sub)))

    return {
        "cluster_assignments": jnp.array(assignments),
        "cluster_sizes": cluster_sizes,
        "within_cluster_correlation": within_corr,
        "between_cluster_correlation": between_corr,
        "n_clusters": n_clusters,
        "correlation_matrix": jnp.array(corr),
    }


def interaction_strength(
    model,
    tokens_list: list,
    metric_fn: Callable,
    components: Optional[list] = None,
) -> dict:
    """Measure interaction strength between model components.

    Tests pairs of components (layer, head) for synergistic or
    redundant interactions by comparing single vs joint ablation.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        metric_fn: fn(logits, tokens) -> scalar.
        components: List of (layer, head) tuples to test. Defaults to all heads
            in layers 0 and 1.

    Returns:
        Dict with interaction_scores, synergistic_pairs, redundant_pairs.
    """
    from irtk.hook_points import HookState

    if components is None:
        components = [(l, h) for l in range(min(2, model.cfg.n_layers))
                      for h in range(model.cfg.n_heads)]

    # Baseline
    baselines = []
    for tok in tokens_list:
        logits = model(tok)
        baselines.append(float(metric_fn(logits, tok)))
    mean_baseline = float(np.mean(baselines))

    # Single ablation effects
    single_effects = {}
    for layer, head in components:
        hook_name = f"blocks.{layer}.attn.hook_result"

        def zero_hook(x, name):
            return jnp.zeros_like(x)

        effects = []
        for tok in tokens_list:
            hook_state = HookState(hook_fns={hook_name: zero_hook}, cache=None)
            logits = model(tok, hook_state=hook_state)
            effects.append(float(metric_fn(logits, tok)))
        single_effects[(layer, head)] = mean_baseline - float(np.mean(effects))

    # Pairwise interactions
    n = len(components)
    scores = np.zeros((n, n))
    synergistic = []
    redundant = []

    for i in range(n):
        scores[i, i] = single_effects[components[i]]
        for j in range(i + 1, n):
            li, hi = components[i]
            lj, hj = components[j]
            hook_i = f"blocks.{li}.attn.hook_result"
            hook_j = f"blocks.{lj}.attn.hook_result"

            def zero_hook(x, name):
                return jnp.zeros_like(x)

            hooks = {hook_i: zero_hook}
            if hook_j != hook_i:
                hooks[hook_j] = zero_hook

            effects = []
            for tok in tokens_list:
                hook_state = HookState(hook_fns=hooks, cache=None)
                logits = model(tok, hook_state=hook_state)
                effects.append(float(metric_fn(logits, tok)))
            joint = mean_baseline - float(np.mean(effects))

            interaction = joint - (single_effects[components[i]] + single_effects[components[j]])
            scores[i, j] = interaction
            scores[j, i] = interaction

            if interaction > 1e-6:
                synergistic.append((components[i], components[j], float(interaction)))
            elif interaction < -1e-6:
                redundant.append((components[i], components[j], float(interaction)))

    return {
        "interaction_scores": jnp.array(scores),
        "single_effects": {str(k): v for k, v in single_effects.items()},
        "synergistic_pairs": sorted(synergistic, key=lambda x: -x[2]),
        "redundant_pairs": sorted(redundant, key=lambda x: x[2]),
        "components": components,
    }


def feature_dependency_graph(
    model,
    tokens_list: list,
    layer: int = -1,
    pos: int = -1,
    threshold: float = 0.3,
) -> dict:
    """Build a dependency graph between activation dimensions.

    Constructs a directed graph where edges indicate potential causal
    dependencies between dimensions based on conditional correlations.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Layer to analyze.
        pos: Position to analyze.
        threshold: Minimum |correlation| for an edge.

    Returns:
        Dict with adjacency_matrix, edges, node_degrees, hub_dimensions.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tok in tokens_list:
        cache = {}
        hook_state = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hook_state)
        activations.append(np.array(cache[hook_name][pos]))
    act_matrix = np.stack(activations)

    # Compute correlation
    d = act_matrix.shape[1]
    stds = np.std(act_matrix, axis=0) + 1e-10
    normed = (act_matrix - act_matrix.mean(axis=0)) / stds
    corr = normed.T @ normed / len(act_matrix)

    # Build adjacency (thresholded absolute correlation)
    adj = np.abs(corr)
    np.fill_diagonal(adj, 0)
    adj[adj < threshold] = 0

    # Extract edges
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if adj[i, j] > 0:
                edges.append((i, j, float(adj[i, j])))

    # Node degrees
    degrees = np.sum(adj > 0, axis=1)

    # Hub dimensions (top by degree)
    hub_idx = np.argsort(degrees)[::-1][:10]
    hubs = [(int(idx), int(degrees[idx])) for idx in hub_idx if degrees[idx] > 0]

    return {
        "adjacency_matrix": jnp.array(adj),
        "edges": edges,
        "n_edges": len(edges),
        "node_degrees": jnp.array(degrees),
        "hub_dimensions": hubs,
        "threshold": threshold,
    }
