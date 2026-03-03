"""Polysemanticity detection and analysis.

Tools for identifying and analyzing neurons/features that respond to
multiple unrelated concepts (polysemanticity), a key challenge in
mechanistic interpretability.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


def polysemanticity_score(
    model: HookedTransformer,
    hook_name: str,
    token_sequences: list,
    dimension: int,
) -> dict:
    """Measure how polysemantic a single activation dimension is.

    Uses activation pattern diversity: a monosemantic feature has a
    single tight cluster of activating contexts, while a polysemantic
    one has multiple distinct clusters.

    Args:
        model: HookedTransformer.
        hook_name: Hook point to analyze.
        token_sequences: List of token arrays.
        dimension: Which dimension to analyze.

    Returns:
        Dict with:
        - "score": polysemanticity score (0 = monosemantic, higher = more poly)
        - "activation_variance": variance of activations
        - "bimodality": bimodality coefficient (>0.555 suggests bimodal)
        - "firing_rate": fraction of positions where feature fires
    """
    all_acts = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook_name])
            if acts.ndim > 1 and dimension < acts.shape[-1]:
                all_acts.extend(acts.reshape(-1, acts.shape[-1])[:, dimension].tolist())

    if not all_acts:
        return {"score": 0.0, "activation_variance": 0.0, "bimodality": 0.0, "firing_rate": 0.0}

    acts = np.array(all_acts)
    variance = float(np.var(acts))
    firing_rate = float(np.mean(acts > 0))

    # Bimodality coefficient: (skewness^2 + 1) / kurtosis_excess
    # Values > 0.555 suggest bimodal distribution
    mean = np.mean(acts)
    std = np.std(acts)
    if std < 1e-10:
        bimodality = 0.0
    else:
        centered = acts - mean
        skewness = float(np.mean(centered ** 3) / (std ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std ** 4))
        bimodality = (skewness ** 2 + 1) / max(kurtosis, 1e-10)

    # Polysemanticity score: combination of bimodality and activation pattern diversity
    # High variance + bimodality + moderate firing rate = likely polysemantic
    score = bimodality * min(firing_rate, 1 - firing_rate) * 4  # peaks at 50% firing

    return {
        "score": float(score),
        "activation_variance": variance,
        "bimodality": bimodality,
        "firing_rate": firing_rate,
    }


def feature_context_clusters(
    model: HookedTransformer,
    hook_name: str,
    token_sequences: list,
    dimension: int,
    n_clusters: int = 3,
    top_k: int = 50,
) -> dict:
    """Cluster the contexts where a feature activates to detect polysemanticity.

    A monosemantic feature will have one dominant cluster; a polysemantic
    feature will have multiple distinct clusters.

    Args:
        model: HookedTransformer.
        hook_name: Hook to analyze.
        token_sequences: Token arrays.
        dimension: Feature dimension.
        n_clusters: Number of clusters.
        top_k: Number of top activations to cluster.

    Returns:
        Dict with:
        - "cluster_assignments": [top_k] cluster ID per example
        - "cluster_sizes": [n_clusters] number of examples per cluster
        - "n_effective_clusters": number of clusters with >10% of examples
        - "activation_values": [top_k] activation values
    """
    examples = []

    for pi, tokens in enumerate(token_sequences):
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name not in cache.cache_dict:
            continue
        acts = np.array(cache.cache_dict[hook_name])
        if acts.ndim < 2 or dimension >= acts.shape[-1]:
            continue

        resid = acts.reshape(-1, acts.shape[-1])
        feat_acts = resid[:, dimension]

        for pos in range(len(feat_acts)):
            if feat_acts[pos] > 0:
                examples.append((float(feat_acts[pos]), resid[pos]))

    if not examples:
        return {"cluster_assignments": np.array([]), "cluster_sizes": np.zeros(n_clusters),
                "n_effective_clusters": 0, "activation_values": np.array([])}

    # Sort by activation and take top-k
    examples.sort(key=lambda x: x[0], reverse=True)
    examples = examples[:top_k]

    act_values = np.array([e[0] for e in examples])
    contexts = np.array([e[1] for e in examples])

    if len(contexts) < n_clusters:
        assignments = np.zeros(len(contexts), dtype=int)
        sizes = np.zeros(n_clusters)
        sizes[0] = len(contexts)
        return {"cluster_assignments": assignments, "cluster_sizes": sizes,
                "n_effective_clusters": 1, "activation_values": act_values}

    # Simple k-means
    rng = np.random.RandomState(42)
    centers_idx = [rng.randint(len(contexts))]
    for _ in range(n_clusters - 1):
        dists = np.min([np.sum((contexts - contexts[c]) ** 2, axis=1) for c in centers_idx], axis=0)
        probs = dists / max(np.sum(dists), 1e-10)
        centers_idx.append(rng.choice(len(contexts), p=probs))

    centers = contexts[centers_idx].copy()
    for _ in range(20):
        dists = np.array([np.sum((contexts - c) ** 2, axis=1) for c in centers])
        assignments = np.argmin(dists, axis=0)
        for k in range(n_clusters):
            members = contexts[assignments == k]
            if len(members) > 0:
                centers[k] = np.mean(members, axis=0)

    sizes = np.array([np.sum(assignments == k) for k in range(n_clusters)])
    n_effective = int(np.sum(sizes > 0.1 * len(contexts)))

    return {
        "cluster_assignments": assignments,
        "cluster_sizes": sizes,
        "n_effective_clusters": n_effective,
        "activation_values": act_values,
    }


def activation_decomposition(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Decompose an activation into its constituent SAE features.

    Shows which features contribute to the activation at a given position,
    revealing whether the representation is a superposition of multiple concepts.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        tokens: Token sequence.
        hook_name: Hook where the SAE was trained.
        pos: Position to decompose (-1 for last).

    Returns:
        Dict with:
        - "active_features": list of (feature_idx, activation) sorted descending
        - "n_active": number of active features
        - "reconstruction_error": L2 error of SAE reconstruction
        - "top_feature_fraction": fraction of activation norm from top feature
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    if hook_name not in cache.cache_dict:
        return {"active_features": [], "n_active": 0,
                "reconstruction_error": 0.0, "top_feature_fraction": 0.0}

    acts = cache.cache_dict[hook_name]
    resolved_pos = pos if pos >= 0 else acts.shape[0] + pos
    act = acts[resolved_pos]  # [d_model]

    # Encode
    feat_acts = np.array(sae.encode(act))  # [n_features]
    active_mask = feat_acts > 0
    n_active = int(np.sum(active_mask))

    # Active features sorted by activation
    active_idx = np.where(active_mask)[0]
    active_vals = feat_acts[active_idx]
    order = np.argsort(active_vals)[::-1]
    active_features = [(int(active_idx[i]), float(active_vals[i])) for i in order]

    # Reconstruction error
    recon = np.array(sae.decode(sae.encode(act)))
    error = float(np.linalg.norm(np.array(act) - recon))

    # Top feature fraction
    act_norm = float(np.linalg.norm(np.array(act)))
    if active_features and act_norm > 1e-10:
        top_feat_idx = active_features[0][0]
        top_feat_dir = np.array(sae.W_dec[top_feat_idx])
        top_feat_contribution = active_features[0][1] * float(np.linalg.norm(top_feat_dir))
        top_fraction = top_feat_contribution / act_norm
    else:
        top_fraction = 0.0

    return {
        "active_features": active_features,
        "n_active": n_active,
        "reconstruction_error": error,
        "top_feature_fraction": float(top_fraction),
    }


def feature_interference_matrix(
    sae: SparseAutoencoder,
    feature_indices: list[int],
) -> dict:
    """Compute interference between SAE features via decoder overlap.

    Features interfere when their decoder directions are not orthogonal,
    meaning one feature's activation affects the reconstruction of another.

    Args:
        sae: Trained SparseAutoencoder.
        feature_indices: Which features to analyze.

    Returns:
        Dict with:
        - "interference_matrix": [n, n] cosine similarity of decoder vectors
        - "max_interference": maximum off-diagonal |cosine similarity|
        - "mean_interference": mean off-diagonal |cosine similarity|
        - "orthogonality_score": 1 - mean_interference (1 = perfectly orthogonal)
    """
    n = len(feature_indices)
    W_dec = np.array(sae.W_dec)  # [n_features, d_model]

    # Extract relevant decoder vectors
    vecs = W_dec[feature_indices]  # [n, d_model]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = vecs / norms

    sim = normed @ normed.T  # [n, n]

    # Off-diagonal statistics
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_diag = np.abs(sim[mask])
        max_inter = float(np.max(off_diag))
        mean_inter = float(np.mean(off_diag))
    else:
        max_inter = 0.0
        mean_inter = 0.0

    return {
        "interference_matrix": sim,
        "max_interference": max_inter,
        "mean_interference": mean_inter,
        "orthogonality_score": 1.0 - mean_inter,
    }


def monosemanticity_ranking(
    model: HookedTransformer,
    hook_name: str,
    token_sequences: list,
    top_k: int = 10,
) -> dict:
    """Rank all dimensions by monosemanticity score.

    Identifies the most monosemantic (interpretable) and most polysemantic
    (mixed) dimensions at a given hook point.

    Args:
        model: HookedTransformer.
        hook_name: Hook point to analyze.
        token_sequences: Token arrays for analysis.
        top_k: How many top/bottom to return.

    Returns:
        Dict with:
        - "most_monosemantic": [(dim_idx, score), ...] most monosemantic dims
        - "most_polysemantic": [(dim_idx, score), ...] most polysemantic dims
        - "all_scores": [d] polysemanticity score for each dimension
        - "mean_score": overall mean polysemanticity
    """
    # Collect activations
    all_acts = []
    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook_name])
            if acts.ndim > 1:
                all_acts.append(acts.reshape(-1, acts.shape[-1]))

    if not all_acts:
        return {"most_monosemantic": [], "most_polysemantic": [],
                "all_scores": np.array([]), "mean_score": 0.0}

    combined = np.concatenate(all_acts, axis=0)
    d = combined.shape[1]

    scores = np.zeros(d)
    for dim in range(d):
        col = combined[:, dim]
        firing_rate = float(np.mean(col > 0))
        std = float(np.std(col))
        mean = float(np.mean(col))

        if std < 1e-10:
            scores[dim] = 0.0
            continue

        centered = col - mean
        skewness = float(np.mean(centered ** 3) / (std ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std ** 4))
        bimodality = (skewness ** 2 + 1) / max(kurtosis, 1e-10)

        scores[dim] = bimodality * min(firing_rate, 1 - firing_rate) * 4

    # Sort
    mono_idx = np.argsort(scores)[:top_k]
    poly_idx = np.argsort(scores)[::-1][:top_k]

    return {
        "most_monosemantic": [(int(i), float(scores[i])) for i in mono_idx],
        "most_polysemantic": [(int(i), float(scores[i])) for i in poly_idx],
        "all_scores": scores,
        "mean_score": float(np.mean(scores)),
    }
