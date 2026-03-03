"""Activation clustering analysis.

Cluster-based analysis of model activations: identify activation archetypes,
transition patterns, and structural groupings in the residual stream.

References:
    Gurnee et al. (2023) "Finding Neurons in a Haystack"
    Bricken et al. (2023) "Towards Monosemanticity"
"""

import jax
import jax.numpy as jnp
import numpy as np


def residual_stream_clustering(model, tokens_list, n_clusters=3, pos=-1):
    """Cluster residual stream activations across inputs at a given position.

    Args:
        model: HookedTransformer model.
        tokens_list: List of input token arrays.
        n_clusters: Number of clusters.
        pos: Position to analyze.

    Returns:
        dict with:
            cluster_assignments: [n_inputs] cluster label per input
            cluster_centers: [n_clusters, d_model] cluster centroids
            cluster_sizes: [n_clusters] number of inputs per cluster
            within_cluster_variance: [n_clusters] variance within each cluster
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Collect final residual activations
    activations = []
    for tokens in tokens_list:
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        key = f"blocks.{n_layers - 1}.hook_resid_post"
        r = cache_state.cache.get(key)
        if r is not None:
            activations.append(np.array(r[pos]))

    if len(activations) == 0:
        return {
            "cluster_assignments": np.zeros(len(tokens_list), dtype=int),
            "cluster_centers": np.zeros((n_clusters, d_model)),
            "cluster_sizes": np.zeros(n_clusters, dtype=int),
            "within_cluster_variance": np.zeros(n_clusters),
        }

    X = np.stack(activations)  # [n_inputs, d_model]
    n_inputs = X.shape[0]
    n_clusters = min(n_clusters, n_inputs)

    # K-means clustering
    rng = np.random.RandomState(42)
    centers = X[rng.choice(n_inputs, n_clusters, replace=False)]

    for _ in range(20):
        # Assign
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=-1)  # [n, k]
        assignments = np.argmin(dists, axis=1)

        # Update
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = assignments == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Compute stats
    sizes = np.zeros(n_clusters, dtype=int)
    variances = np.zeros(n_clusters)
    for k in range(n_clusters):
        mask = assignments == k
        sizes[k] = int(np.sum(mask))
        if sizes[k] > 0:
            variances[k] = float(np.mean(np.var(X[mask], axis=0)))

    return {
        "cluster_assignments": assignments,
        "cluster_centers": centers,
        "cluster_sizes": sizes,
        "within_cluster_variance": variances,
    }


def layer_activation_archetypes(model, tokens, n_archetypes=3):
    """Identify activation archetypes across layers.

    Clusters residual stream activations at each position across all layers
    to find recurring activation patterns (archetypes).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        n_archetypes: Number of archetypes to find.

    Returns:
        dict with:
            archetypes: [n_archetypes, d_model] archetype vectors
            layer_archetype_assignments: [n_layers, seq_len] which archetype per layer/position
            archetype_prevalence: [n_archetypes] fraction of activations in each archetype
            layer_archetype_distribution: [n_layers, n_archetypes] distribution per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    # Collect all activations
    all_acts = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        r = cache.get(key)
        if r is not None:
            all_acts.append(np.array(r))  # [seq_len, d_model]

    if len(all_acts) == 0:
        return {
            "archetypes": np.zeros((n_archetypes, d_model)),
            "layer_archetype_assignments": np.zeros((n_layers, seq_len), dtype=int),
            "archetype_prevalence": np.zeros(n_archetypes),
            "layer_archetype_distribution": np.zeros((n_layers, n_archetypes)),
        }

    X = np.concatenate(all_acts, axis=0)  # [n_layers * seq_len, d_model]
    n_total = X.shape[0]
    n_archetypes = min(n_archetypes, n_total)

    # K-means
    rng = np.random.RandomState(42)
    centers = X[rng.choice(n_total, n_archetypes, replace=False)]

    for _ in range(20):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=-1)
        assignments = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(n_archetypes):
            mask = assignments == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Reshape assignments
    layer_assignments = assignments.reshape(n_layers, seq_len)

    # Prevalence
    prevalence = np.zeros(n_archetypes)
    for k in range(n_archetypes):
        prevalence[k] = float(np.mean(assignments == k))

    # Layer distribution
    layer_dist = np.zeros((n_layers, n_archetypes))
    for l in range(n_layers):
        for k in range(n_archetypes):
            layer_dist[l, k] = float(np.mean(layer_assignments[l] == k))

    return {
        "archetypes": centers,
        "layer_archetype_assignments": layer_assignments,
        "archetype_prevalence": prevalence,
        "layer_archetype_distribution": layer_dist,
    }


def activation_transition_analysis(model, tokens, pos=-1):
    """Analyze how activations transition between layers.

    Measures the magnitude and direction of transitions at each layer step.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            transition_magnitudes: [n_layers-1] norm of transition at each step
            transition_directions: [n_layers-1, d_model] normalized transition vectors
            cosine_continuity: [n_layers-2] cosine sim between consecutive transitions
            mean_transition_magnitude: float
            smoothness: float (mean cosine continuity)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    resids = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        r = cache.get(key)
        if r is not None:
            resids.append(np.array(r[pos]))

    if len(resids) < 2:
        d_model = model.cfg.d_model
        return {
            "transition_magnitudes": np.zeros(max(0, n_layers - 1)),
            "transition_directions": np.zeros((max(0, n_layers - 1), d_model)),
            "cosine_continuity": np.zeros(max(0, n_layers - 2)),
            "mean_transition_magnitude": 0.0,
            "smoothness": 0.0,
        }

    # Transitions
    n = len(resids)
    d_model = resids[0].shape[0]
    magnitudes = np.zeros(n - 1)
    directions = np.zeros((n - 1, d_model))

    for i in range(n - 1):
        delta = resids[i + 1] - resids[i]
        mag = float(np.linalg.norm(delta))
        magnitudes[i] = mag
        if mag > 1e-10:
            directions[i] = delta / mag

    # Cosine continuity
    cosines = np.zeros(max(0, n - 2))
    for i in range(n - 2):
        dot = float(np.dot(directions[i], directions[i + 1]))
        cosines[i] = dot

    return {
        "transition_magnitudes": magnitudes,
        "transition_directions": directions,
        "cosine_continuity": cosines,
        "mean_transition_magnitude": float(np.mean(magnitudes)),
        "smoothness": float(np.mean(cosines)) if len(cosines) > 0 else 0.0,
    }


def position_clustering(model, tokens, layer=-1, n_clusters=3):
    """Cluster token positions based on their activations at a given layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer to analyze (-1 for last).
        n_clusters: Number of clusters.

    Returns:
        dict with:
            cluster_assignments: [seq_len] cluster label per position
            cluster_centers: [n_clusters, d_model] cluster centroids
            cluster_sizes: [n_clusters] positions per cluster
            position_similarity: [seq_len, seq_len] pairwise cosine similarity
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if layer < 0:
        layer = n_layers + layer

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    key = f"blocks.{layer}.hook_resid_post"
    r = cache_state.cache.get(key)

    seq_len = len(tokens)
    d_model = model.cfg.d_model

    if r is None:
        return {
            "cluster_assignments": np.zeros(seq_len, dtype=int),
            "cluster_centers": np.zeros((n_clusters, d_model)),
            "cluster_sizes": np.zeros(n_clusters, dtype=int),
            "position_similarity": np.eye(seq_len),
        }

    X = np.array(r)  # [seq_len, d_model]
    n_clusters = min(n_clusters, seq_len)

    # K-means
    rng = np.random.RandomState(42)
    centers = X[rng.choice(seq_len, n_clusters, replace=False)]

    for _ in range(20):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=-1)
        assignments = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = assignments == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    sizes = np.zeros(n_clusters, dtype=int)
    for k in range(n_clusters):
        sizes[k] = int(np.sum(assignments == k))

    # Pairwise cosine similarity
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_normed = X / norms
    sim = X_normed @ X_normed.T

    return {
        "cluster_assignments": assignments,
        "cluster_centers": centers,
        "cluster_sizes": sizes,
        "position_similarity": sim,
    }


def component_output_clustering(model, tokens, n_clusters=3, pos=-1):
    """Cluster component outputs (attn, MLP per layer) by similarity.

    Groups components that produce similar residual stream updates.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        n_clusters: Number of clusters.
        pos: Position to analyze.

    Returns:
        dict with:
            component_names: list of component names
            cluster_assignments: [n_components] cluster per component
            cluster_centers: [n_clusters, d_model]
            similarity_matrix: [n_components, n_components] cosine similarity
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    names = []
    vecs = []

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn is not None:
            names.append(f"attn_L{layer}")
            vecs.append(np.array(attn[pos]))

        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp is not None:
            names.append(f"mlp_L{layer}")
            vecs.append(np.array(mlp[pos]))

    if len(vecs) == 0:
        return {
            "component_names": [],
            "cluster_assignments": np.array([], dtype=int),
            "cluster_centers": np.zeros((n_clusters, d_model)),
            "similarity_matrix": np.array([[]]),
        }

    X = np.stack(vecs)  # [n_components, d_model]
    n_comp = X.shape[0]
    n_clusters = min(n_clusters, n_comp)

    # K-means
    rng = np.random.RandomState(42)
    centers = X[rng.choice(n_comp, n_clusters, replace=False)]

    for _ in range(20):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=-1)
        assignments = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = assignments == k
            if np.any(mask):
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Similarity matrix
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_normed = X / norms
    sim = X_normed @ X_normed.T

    return {
        "component_names": names,
        "cluster_assignments": assignments,
        "cluster_centers": centers,
        "similarity_matrix": sim,
    }
