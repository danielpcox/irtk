"""SAE feature dictionary geometry analysis.

Analyze population-level properties of SAE feature dictionaries:
feature splitting across scales, absorption detection, universality
across models, co-occurrence structure, and decoder geometry.

References:
    - Bricken et al. (2023) "Towards Monosemanticity"
    - Templeton et al. (2024) "Scaling Monosemanticity"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


def feature_splitting_analysis(
    sae_small: SparseAutoencoder,
    sae_large: SparseAutoencoder,
    model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    cosine_threshold: float = 0.5,
) -> dict:
    """Compare feature geometry between two SAEs of different widths.

    Identifies which small-SAE features split into multiple large-SAE
    features by matching decoder vectors and activation correlations.

    Args:
        sae_small: Narrower SAE.
        sae_large: Wider SAE.
        model: HookedTransformer for collecting activations.
        token_sequences: Test inputs.
        hook_name: Hook point for activations.
        cosine_threshold: Minimum cosine to count as a match.

    Returns:
        Dict with:
        - "split_features": list of (small_idx, [large_idxs], max_cosine)
        - "unsplit_features": small feature indices with a single match
        - "mean_split_count": average number of large features per small feature
        - "decoder_cosines": [n_small, n_large] cosine similarity matrix
    """
    W_small = np.array(sae_small.W_dec)  # [n_small, d_model]
    W_large = np.array(sae_large.W_dec)  # [n_large, d_model]

    # Normalize rows
    norm_s = np.linalg.norm(W_small, axis=-1, keepdims=True)
    norm_l = np.linalg.norm(W_large, axis=-1, keepdims=True)
    W_s_n = W_small / np.maximum(norm_s, 1e-10)
    W_l_n = W_large / np.maximum(norm_l, 1e-10)

    # Cosine similarity matrix
    cosines = W_s_n @ W_l_n.T  # [n_small, n_large]

    split_features = []
    unsplit_features = []

    for si in range(sae_small.n_features):
        matches = list(np.where(cosines[si] > cosine_threshold)[0])
        max_cos = float(np.max(cosines[si])) if len(cosines[si]) > 0 else 0.0
        if len(matches) > 1:
            split_features.append((si, matches, max_cos))
        else:
            unsplit_features.append(si)

    split_counts = [len(m) for _, m, _ in split_features]
    mean_split = float(np.mean(split_counts)) if split_counts else 1.0

    return {
        "split_features": split_features,
        "unsplit_features": unsplit_features,
        "mean_split_count": mean_split,
        "decoder_cosines": cosines,
    }


def feature_absorption_detection(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    feature_a: int,
    feature_b: int,
) -> dict:
    """Test whether feature_b absorbs feature_a's role in its presence.

    Absorption: feature_a fires when feature_b is absent, but when
    feature_b is present, feature_a goes dark.

    Args:
        sae: SparseAutoencoder.
        model: HookedTransformer.
        token_sequences: Test inputs.
        hook_name: Hook point.
        feature_a: Feature that may be absorbed.
        feature_b: Feature that may absorb.

    Returns:
        Dict with:
        - "absorption_score": how strongly B suppresses A (0-1)
        - "a_rate_without_b": firing rate of A when B is inactive
        - "a_rate_with_b": firing rate of A when B is active
        - "suppression_ratio": a_rate_with_b / a_rate_without_b
    """
    a_with_b = []
    a_without_b = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name not in cache.cache_dict:
            continue

        acts = cache.cache_dict[hook_name]
        flat = acts.reshape(-1, acts.shape[-1])
        feats = np.array(sae.encode(flat))  # [n_positions, n_features]

        for pos in range(feats.shape[0]):
            a_val = feats[pos, feature_a]
            b_val = feats[pos, feature_b]
            if b_val > 0:
                a_with_b.append(float(a_val > 0))
            else:
                a_without_b.append(float(a_val > 0))

    rate_with = float(np.mean(a_with_b)) if a_with_b else 0.0
    rate_without = float(np.mean(a_without_b)) if a_without_b else 0.0

    if rate_without > 1e-10:
        suppression = rate_with / rate_without
    else:
        suppression = 1.0

    absorption = max(0.0, 1.0 - suppression)

    return {
        "absorption_score": float(absorption),
        "a_rate_without_b": rate_without,
        "a_rate_with_b": rate_with,
        "suppression_ratio": float(suppression),
    }


def feature_universality(
    sae_a: SparseAutoencoder,
    sae_b: SparseAutoencoder,
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    top_k: int = 50,
) -> dict:
    """Measure feature universality across two SAEs on different models.

    Matches features by activation correlation across shared inputs.

    Args:
        sae_a: SAE for model A.
        sae_b: SAE for model B.
        model_a: First model.
        model_b: Second model.
        token_sequences: Shared test inputs.
        hook_name: Hook point.
        top_k: Number of top features to consider.

    Returns:
        Dict with:
        - "matched_pairs": list of (feat_a, feat_b, correlation)
        - "universality_rate": fraction of A's features matched in B
        - "mean_match_correlation": average correlation of best matches
        - "decoder_cosines": [min(top_k, n_a), min(top_k, n_b)] decoder cosine matrix
    """
    # Collect activations
    feats_a_all = []
    feats_b_all = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)

        _, cache_a = model_a.run_with_cache(tokens)
        _, cache_b = model_b.run_with_cache(tokens)

        if hook_name not in cache_a.cache_dict or hook_name not in cache_b.cache_dict:
            continue

        acts_a = cache_a.cache_dict[hook_name]
        acts_b = cache_b.cache_dict[hook_name]

        flat_a = acts_a.reshape(-1, acts_a.shape[-1])
        flat_b = acts_b.reshape(-1, acts_b.shape[-1])

        fa = np.array(sae_a.encode(flat_a))  # [pos, n_features_a]
        fb = np.array(sae_b.encode(flat_b))  # [pos, n_features_b]

        min_pos = min(fa.shape[0], fb.shape[0])
        feats_a_all.append(fa[:min_pos])
        feats_b_all.append(fb[:min_pos])

    if not feats_a_all:
        return {"matched_pairs": [], "universality_rate": 0.0,
                "mean_match_correlation": 0.0, "decoder_cosines": np.array([])}

    all_a = np.concatenate(feats_a_all, axis=0)  # [N, n_a]
    all_b = np.concatenate(feats_b_all, axis=0)  # [N, n_b]

    # Use top_k most active features
    n_a = min(top_k, all_a.shape[1])
    n_b = min(top_k, all_b.shape[1])
    top_a = np.argsort(np.mean(all_a, axis=0))[-n_a:]
    top_b = np.argsort(np.mean(all_b, axis=0))[-n_b:]

    a_sub = all_a[:, top_a]  # [N, n_a]
    b_sub = all_b[:, top_b]  # [N, n_b]

    # Correlation matrix
    corr = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            va, vb = a_sub[:, i], b_sub[:, j]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 1e-10 and nb > 1e-10:
                corr[i, j] = float(np.dot(va, vb) / (na * nb))

    # Best matches
    matched = []
    match_corrs = []
    for i in range(n_a):
        best_j = int(np.argmax(corr[i]))
        best_c = float(corr[i, best_j])
        matched.append((int(top_a[i]), int(top_b[best_j]), best_c))
        match_corrs.append(best_c)

    matched.sort(key=lambda x: x[2], reverse=True)
    n_matched = sum(1 for _, _, c in matched if c > 0.5)
    universality = n_matched / max(n_a, 1)

    # Decoder cosines
    W_a = np.array(sae_a.W_dec[top_a])
    W_b = np.array(sae_b.W_dec[top_b])
    na_w = np.linalg.norm(W_a, axis=-1, keepdims=True)
    nb_w = np.linalg.norm(W_b, axis=-1, keepdims=True)
    min_d = min(W_a.shape[-1], W_b.shape[-1])
    dec_cos = (W_a[:, :min_d] / np.maximum(na_w, 1e-10)) @ (W_b[:, :min_d] / np.maximum(nb_w, 1e-10)).T

    return {
        "matched_pairs": matched,
        "universality_rate": float(universality),
        "mean_match_correlation": float(np.mean(match_corrs)) if match_corrs else 0.0,
        "decoder_cosines": dec_cos,
    }


def feature_interaction_graph(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    top_k: int = 50,
) -> dict:
    """Build a graph of feature co-occurrence and suppression relationships.

    Args:
        sae: SparseAutoencoder.
        model: HookedTransformer.
        token_sequences: Test inputs.
        hook_name: Hook point.
        top_k: Number of most active features to analyze.

    Returns:
        Dict with:
        - "co_occurrence": [top_k, top_k] co-activation rate matrix
        - "correlations": [top_k, top_k] activation correlation matrix
        - "suppression_pairs": list of (feat_i, feat_j) with negative correlation
        - "feature_indices": the top_k feature indices used
    """
    all_feats = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name not in cache.cache_dict:
            continue
        acts = cache.cache_dict[hook_name]
        flat = acts.reshape(-1, acts.shape[-1])
        feats = np.array(sae.encode(flat))
        all_feats.append(feats)

    if not all_feats:
        return {"co_occurrence": np.array([]), "correlations": np.array([]),
                "suppression_pairs": [], "feature_indices": []}

    all_f = np.concatenate(all_feats, axis=0)  # [N, n_features]
    n_f = min(top_k, all_f.shape[1])
    top_idx = np.argsort(np.mean(all_f, axis=0))[-n_f:]
    sub = all_f[:, top_idx]  # [N, n_f]

    # Binary co-occurrence
    active = (sub > 0).astype(float)
    co_occ = (active.T @ active) / max(active.shape[0], 1)

    # Correlation
    corr = np.zeros((n_f, n_f))
    for i in range(n_f):
        for j in range(n_f):
            vi, vj = sub[:, i], sub[:, j]
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni > 1e-10 and nj > 1e-10:
                corr[i, j] = float(np.dot(vi, vj) / (ni * nj))

    # Suppression: pairs with negative correlation
    suppression = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            if corr[i, j] < -0.1:
                suppression.append((int(top_idx[i]), int(top_idx[j])))

    return {
        "co_occurrence": co_occ,
        "correlations": corr,
        "suppression_pairs": suppression,
        "feature_indices": top_idx.tolist(),
    }


def decoder_geometry_stats(
    sae: SparseAutoencoder,
) -> dict:
    """Analyze the geometry of the decoder dictionary matrix.

    Computes pairwise cosine similarities, near-duplicates, feature norms.

    Args:
        sae: SparseAutoencoder.

    Returns:
        Dict with:
        - "pairwise_cosines": [n_features, n_features] cosine similarity matrix
        - "near_duplicates": list of (i, j) pairs with cosine > 0.9
        - "feature_norms": [n_features] L2 norms of decoder vectors
        - "mean_pairwise_cosine": mean off-diagonal cosine
        - "n_near_duplicates": count of near-duplicate pairs
    """
    W = np.array(sae.W_dec)  # [n_features, d_model]
    norms = np.linalg.norm(W, axis=-1)  # [n_features]
    W_normed = W / np.maximum(norms[:, None], 1e-10)

    cosines = W_normed @ W_normed.T  # [n, n]

    # Near duplicates
    near_dups = []
    n = cosines.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if cosines[i, j] > 0.9:
                near_dups.append((i, j))

    # Mean off-diagonal cosine
    mask = ~np.eye(n, dtype=bool)
    mean_cos = float(np.mean(cosines[mask])) if n > 1 else 0.0

    return {
        "pairwise_cosines": cosines,
        "near_duplicates": near_dups,
        "feature_norms": norms,
        "mean_pairwise_cosine": mean_cos,
        "n_near_duplicates": len(near_dups),
    }
