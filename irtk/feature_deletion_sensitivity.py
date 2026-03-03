"""Feature deletion sensitivity analysis.

Measures model sensitivity to deletion of individual semantic features
or high-level concepts. Bridges interpretability from granular components
to human-understandable concepts through systematic deletion analysis.

Functions:
- feature_importance_ranking: Rank features by impact on output
- feature_deletion_cascade: Iteratively delete features and track degradation
- feature_interaction_effects: Measure interaction between feature pairs
- minimal_sufficient_features: Find smallest feature set preserving performance
- feature_redundancy_clusters: Group features by functional redundancy

References:
    - Bricken et al. (2023) "Towards Monosemanticity"
    - Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features"
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def feature_importance_ranking(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_directions: np.ndarray,
    metric_fn: Callable,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Rank features by their impact on the output.

    For each feature direction, measures the effect of removing that
    direction from the residual stream.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        feature_directions: [n_features, d_model] feature directions.
        metric_fn: Function(logits) -> float.
        layer: Layer at which to intervene (-1 = last).
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "importance_scores": [n_features] absolute metric change from deletion
            "ranking": [n_features] indices sorted by importance (most important first)
            "baseline_metric": original metric
            "top_feature": index of most important feature
    """
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers
    if layer == -1:
        layer = n_layers - 1
    if pos == -1:
        pos = seq_len - 1

    baseline_logits = np.array(model(tokens))
    baseline = float(metric_fn(baseline_logits))

    n_features = len(feature_directions)
    scores = np.zeros(n_features)

    for i in range(n_features):
        direction = feature_directions[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        unit_dir = direction / norm

        # Hook to remove this direction
        dir_jax = jnp.array(unit_dir)
        actual_pos = pos

        def make_hook(d):
            def hook_fn(x, name):
                proj = jnp.dot(x[actual_pos], d) * d
                return x.at[actual_pos].add(-proj)
            return hook_fn

        hook_name = f"blocks.{layer}.hook_resid_post"
        try:
            patched_logits = np.array(
                model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(dir_jax))])
            )
            scores[i] = abs(float(metric_fn(patched_logits)) - baseline)
        except Exception:
            scores[i] = 0.0

    ranking = np.argsort(scores)[::-1]
    top = int(ranking[0]) if len(ranking) > 0 else 0

    return {
        "importance_scores": scores,
        "ranking": ranking,
        "baseline_metric": baseline,
        "top_feature": top,
    }


def feature_deletion_cascade(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_directions: np.ndarray,
    metric_fn: Callable,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Iteratively delete features and track metric degradation.

    Deletes features one at a time in order of importance, tracking
    cumulative metric change.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        feature_directions: [n_features, d_model] feature directions.
        metric_fn: Function(logits) -> float.
        layer: Layer (-1 = last).
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "deletion_order": [n_features] order of feature deletion
            "cumulative_metrics": [n_features+1] metric after each deletion
            "metric_at_half": metric after deleting half the features
            "features_for_90pct": number of features needed to drop metric by 90%
    """
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers
    if layer == -1:
        layer = n_layers - 1
    if pos == -1:
        pos = seq_len - 1

    # Get importance ranking
    ranking_result = feature_importance_ranking(
        model, tokens, feature_directions, metric_fn, layer, pos
    )
    order = ranking_result["ranking"]
    baseline = ranking_result["baseline_metric"]

    n_features = len(feature_directions)
    metrics = [baseline]

    # Build cumulative deletion projection
    deleted_dirs = []
    for step in range(n_features):
        feat_idx = int(order[step])
        direction = feature_directions[feat_idx]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            deleted_dirs.append(direction / norm)

        # Hook to remove all deleted directions
        dirs_jax = [jnp.array(d) for d in deleted_dirs]
        actual_pos = pos

        def make_hook(dirs):
            def hook_fn(x, name):
                result = x
                for d in dirs:
                    proj = jnp.dot(result[actual_pos], d) * d
                    result = result.at[actual_pos].add(-proj)
                return result
            return hook_fn

        hook_name = f"blocks.{layer}.hook_resid_post"
        try:
            patched_logits = np.array(
                model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(dirs_jax))])
            )
            metrics.append(float(metric_fn(patched_logits)))
        except Exception:
            metrics.append(metrics[-1])

    metrics = np.array(metrics)

    # Metric at half
    half_idx = min(n_features // 2 + 1, len(metrics) - 1)
    metric_half = float(metrics[half_idx])

    # Features for 90% drop
    target = baseline * 0.1 if baseline > 0 else baseline * 10
    features_90 = n_features
    for i in range(len(metrics)):
        if abs(metrics[i]) < abs(target):
            features_90 = i
            break

    return {
        "deletion_order": order,
        "cumulative_metrics": metrics,
        "metric_at_half": metric_half,
        "features_for_90pct": features_90,
    }


def feature_interaction_effects(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_directions: np.ndarray,
    metric_fn: Callable,
    feature_pairs: Optional[list] = None,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Measure interaction between feature pairs.

    Compares individual deletion effects vs combined deletion to detect
    synergy or redundancy.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        feature_directions: [n_features, d_model] feature directions.
        metric_fn: Function(logits) -> float.
        feature_pairs: list of (i, j) pairs to test. Default: all pairs for small n.
        layer: Layer (-1 = last).
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "interaction_matrix": [n_pairs] interaction score (positive = synergy)
            "pair_indices": list of (i, j) pairs tested
            "strongest_interaction": (i, j) with largest absolute interaction
            "mean_interaction": mean absolute interaction score
    """
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers
    n_features = len(feature_directions)
    if layer == -1:
        layer = n_layers - 1
    if pos == -1:
        pos = seq_len - 1

    baseline_logits = np.array(model(tokens))
    baseline = float(metric_fn(baseline_logits))

    if feature_pairs is None:
        feature_pairs = [(i, j) for i in range(min(n_features, 5)) for j in range(i+1, min(n_features, 5))]

    interactions = []
    actual_pos = pos

    for i, j in feature_pairs:
        # Individual effects
        effects = []
        for feat_idx in [i, j]:
            d = feature_directions[feat_idx]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                effects.append(0.0)
                continue
            unit = d / norm
            d_jax = jnp.array(unit)

            def make_hook(direction):
                def hook_fn(x, name):
                    proj = jnp.dot(x[actual_pos], direction) * direction
                    return x.at[actual_pos].add(-proj)
                return hook_fn

            hook_name = f"blocks.{layer}.hook_resid_post"
            try:
                patched = np.array(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(d_jax))]))
                effects.append(abs(float(metric_fn(patched)) - baseline))
            except Exception:
                effects.append(0.0)

        # Combined effect
        dirs = []
        for feat_idx in [i, j]:
            d = feature_directions[feat_idx]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                dirs.append(jnp.array(d / norm))

        def make_combined_hook(directions):
            def hook_fn(x, name):
                result = x
                for d in directions:
                    proj = jnp.dot(result[actual_pos], d) * d
                    result = result.at[actual_pos].add(-proj)
                return result
            return hook_fn

        try:
            patched = np.array(
                model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_combined_hook(dirs))])
            )
            combined = abs(float(metric_fn(patched)) - baseline)
        except Exception:
            combined = sum(effects)

        # Interaction = combined - (individual_i + individual_j)
        interaction = combined - sum(effects)
        interactions.append(interaction)

    interactions = np.array(interactions)
    strongest_idx = int(np.argmax(np.abs(interactions))) if len(interactions) > 0 else 0
    strongest = feature_pairs[strongest_idx] if feature_pairs else (0, 0)

    return {
        "interaction_matrix": interactions,
        "pair_indices": feature_pairs,
        "strongest_interaction": strongest,
        "mean_interaction": float(np.mean(np.abs(interactions))) if len(interactions) > 0 else 0.0,
    }


def minimal_sufficient_features(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_directions: np.ndarray,
    metric_fn: Callable,
    threshold: float = 0.9,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Find smallest feature set preserving performance.

    Greedily adds features until the threshold fraction of the baseline
    metric is reached.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        feature_directions: [n_features, d_model] feature directions.
        metric_fn: Function(logits) -> float.
        threshold: Fraction of baseline metric to preserve.
        layer: Layer (-1 = last).
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "sufficient_features": list of feature indices in the minimal set
            "n_sufficient": number of features needed
            "fraction_of_total": fraction of all features needed
            "achieved_metric": metric with only the sufficient features
    """
    cascade = feature_deletion_cascade(
        model, tokens, feature_directions, metric_fn, layer, pos
    )

    baseline = cascade["cumulative_metrics"][0]
    n_features = len(feature_directions)
    target = abs(baseline * threshold)

    # From cascade, find how many features we can delete before dropping below threshold
    sufficient = list(range(n_features))  # start with all
    for step in range(len(cascade["cumulative_metrics"]) - 1):
        if abs(cascade["cumulative_metrics"][step + 1]) >= target:
            # We can delete this feature
            feat_idx = int(cascade["deletion_order"][step])
            if feat_idx in sufficient:
                sufficient.remove(feat_idx)
        else:
            break

    achieved = float(cascade["cumulative_metrics"][min(n_features - len(sufficient), len(cascade["cumulative_metrics"]) - 1)])

    return {
        "sufficient_features": sufficient,
        "n_sufficient": len(sufficient),
        "fraction_of_total": len(sufficient) / max(n_features, 1),
        "achieved_metric": achieved,
    }


def feature_redundancy_clusters(
    model: HookedTransformer,
    feature_directions: np.ndarray,
    n_clusters: int = 3,
) -> dict:
    """Group features by geometric similarity.

    Clusters feature directions by cosine similarity to find groups
    of redundant or related features.

    Args:
        model: HookedTransformer (for d_model).
        feature_directions: [n_features, d_model] directions.
        n_clusters: Number of clusters.

    Returns:
        Dict with:
            "cluster_assignments": [n_features] cluster index per feature
            "cluster_sizes": [n_clusters] number of features per cluster
            "within_cluster_similarity": [n_clusters] mean cosine similarity within
            "between_cluster_similarity": mean cosine similarity between clusters
    """
    n_features = len(feature_directions)
    n_clusters = min(n_clusters, n_features)

    # Compute similarity matrix
    norms = np.linalg.norm(feature_directions, axis=1, keepdims=True) + 1e-10
    normed = feature_directions / norms
    sim_matrix = normed @ normed.T  # [n_features, n_features]

    # Simple k-means-style clustering on cosine similarity
    # Initialize with spread-out features
    assignments = np.zeros(n_features, dtype=int)
    for i in range(n_features):
        assignments[i] = i % n_clusters

    # Iterate a few times
    for _ in range(10):
        # Update centroids
        centroids = np.zeros((n_clusters, feature_directions.shape[1]))
        for c in range(n_clusters):
            mask = assignments == c
            if np.any(mask):
                centroids[c] = np.mean(normed[mask], axis=0)
                cn = np.linalg.norm(centroids[c])
                if cn > 1e-10:
                    centroids[c] /= cn

        # Reassign
        for i in range(n_features):
            best_sim = -2.0
            best_c = 0
            for c in range(n_clusters):
                s = float(np.dot(normed[i], centroids[c]))
                if s > best_sim:
                    best_sim = s
                    best_c = c
            assignments[i] = best_c

    # Compute stats
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    within_sim = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = assignments == c
        cluster_sizes[c] = int(np.sum(mask))
        if cluster_sizes[c] > 1:
            cluster_features = normed[mask]
            pair_sims = []
            for i in range(len(cluster_features)):
                for j in range(i+1, len(cluster_features)):
                    pair_sims.append(float(np.dot(cluster_features[i], cluster_features[j])))
            within_sim[c] = float(np.mean(pair_sims)) if pair_sims else 1.0
        else:
            within_sim[c] = 1.0

    # Between-cluster similarity
    between_sims = []
    for c1 in range(n_clusters):
        for c2 in range(c1+1, n_clusters):
            mask1 = assignments == c1
            mask2 = assignments == c2
            if np.any(mask1) and np.any(mask2):
                cross = normed[mask1] @ normed[mask2].T
                between_sims.append(float(np.mean(cross)))

    between = float(np.mean(between_sims)) if between_sims else 0.0

    return {
        "cluster_assignments": assignments,
        "cluster_sizes": cluster_sizes,
        "within_cluster_similarity": within_sim,
        "between_cluster_similarity": between,
    }
