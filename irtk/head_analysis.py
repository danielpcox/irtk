"""Head-level analysis toolkit.

Tools for classifying, scoring, and clustering attention heads:
- classify_heads: Classify heads by behavior (previous-token, induction, etc.)
- head_importance_scores: Multiple importance metrics per head
- head_clustering: Cluster heads by behavioral similarity
- find_induction_heads, find_previous_token_heads: Specialized finders
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def _get_attention_pattern(cache: ActivationCache, layer: int, head: int) -> np.ndarray:
    """Extract attention pattern from cache."""
    return np.array(cache[("pattern", layer)][head])


def find_previous_token_heads(
    cache: ActivationCache,
    model: HookedTransformer,
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Find heads that primarily attend to the previous token.

    A previous-token head has most of its attention weight on position i-1
    when querying from position i.

    Args:
        cache: ActivationCache from run_with_cache.
        model: HookedTransformer.
        threshold: Minimum average previous-token attention to qualify.

    Returns:
        List of (layer, head, score) tuples sorted by score descending.
    """
    results = []
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            pattern = _get_attention_pattern(cache, l, h)  # [q, k]
            seq_len = pattern.shape[0]
            if seq_len < 2:
                continue

            # Score: average attention to position i-1 from position i
            prev_attn = np.array([pattern[i, i - 1] for i in range(1, seq_len)])
            score = float(np.mean(prev_attn))
            if score >= threshold:
                results.append((l, h, score))

    return sorted(results, key=lambda x: x[2], reverse=True)


def find_induction_heads(
    model: HookedTransformer,
    seq_len: int = 50,
    threshold: float = 0.5,
    seed: int = 0,
) -> list[tuple[int, int, float]]:
    """Find induction heads using repeated random tokens.

    An induction head attends to the token after the previous occurrence
    of the current query token, implementing the [A][B]...[A] -> [B] pattern.

    Args:
        model: HookedTransformer.
        seq_len: Half-length of the repeated sequence.
        threshold: Minimum induction score to qualify.
        seed: Random seed.

    Returns:
        List of (layer, head, score) tuples sorted by score descending.
    """
    # Create repeated tokens: [random_seq] + [random_seq]
    rng = np.random.RandomState(seed)
    half = rng.randint(1, model.cfg.d_vocab, seq_len)
    tokens = jnp.array(np.concatenate([half, half]))

    _, cache = model.run_with_cache(tokens)

    results = []
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            pattern = _get_attention_pattern(cache, l, h)

            # For positions in the second half, check if they attend
            # to position (i - seq_len + 1) — the token after the
            # previous occurrence
            scores = []
            for i in range(seq_len + 1, 2 * seq_len):
                target = i - seq_len + 1  # next token after previous occurrence
                if target < pattern.shape[1]:
                    scores.append(pattern[i, target])

            if scores:
                score = float(np.mean(scores))
                if score >= threshold:
                    results.append((l, h, score))

    return sorted(results, key=lambda x: x[2], reverse=True)


def head_importance_scores(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Compute multiple importance metrics for every attention head.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: If given, compute direct logit attribution for this token.

    Returns:
        Dict with [n_layers, n_heads] arrays:
        - "entropy": Mean attention entropy
        - "max_attn": Mean of max attention weight
        - "prev_token": Previous-token attention score
        - "diag_score": Self-attention (diagonal) score
        - "direct_logit": Direct logit attribution (if target_token given)
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    entropy_scores = np.zeros((n_layers, n_heads))
    max_attn_scores = np.zeros((n_layers, n_heads))
    prev_token_scores = np.zeros((n_layers, n_heads))
    diag_scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            pattern = _get_attention_pattern(cache, l, h)
            seq_len = pattern.shape[0]

            # Entropy
            clipped = np.clip(pattern, 1e-10, 1.0)
            ent = -np.sum(clipped * np.log(clipped), axis=-1)
            entropy_scores[l, h] = float(np.mean(ent))

            # Max attention
            max_attn_scores[l, h] = float(np.mean(np.max(pattern, axis=-1)))

            # Previous token
            if seq_len > 1:
                prev = [pattern[i, i - 1] for i in range(1, seq_len)]
                prev_token_scores[l, h] = float(np.mean(prev))

            # Diagonal
            diag = [pattern[i, i] for i in range(seq_len)]
            diag_scores[l, h] = float(np.mean(diag))

    result = {
        "entropy": entropy_scores,
        "max_attn": max_attn_scores,
        "prev_token": prev_token_scores,
        "diag_score": diag_scores,
    }

    # Direct logit attribution
    if target_token is not None:
        from irtk.circuits import direct_logit_attribution
        dla = direct_logit_attribution(model, cache, token=target_token)
        result["direct_logit"] = np.array(dla)

    return result


def classify_heads(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    prev_token_threshold: float = 0.4,
    induction_threshold: float = 0.3,
    diagonal_threshold: float = 0.4,
) -> dict[str, list[tuple[int, int]]]:
    """Classify attention heads by their behavior pattern.

    Categories:
    - "previous_token": Primarily attends to position i-1
    - "self_attention": Primarily attends to position i (diagonal)
    - "bos_attention": Primarily attends to position 0 (BOS/first token)
    - "broad": High entropy (attends broadly)
    - "other": Doesn't fit the above categories

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        prev_token_threshold: Threshold for previous-token classification.
        induction_threshold: Threshold for induction classification.
        diagonal_threshold: Threshold for self-attention classification.

    Returns:
        Dict mapping category -> list of (layer, head) tuples.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    categories = {
        "previous_token": [],
        "self_attention": [],
        "bos_attention": [],
        "broad": [],
        "other": [],
    }

    for l in range(n_layers):
        for h in range(n_heads):
            pattern = _get_attention_pattern(cache, l, h)
            seq_len = pattern.shape[0]

            # Compute scores
            prev_score = 0.0
            diag_score = 0.0
            bos_score = 0.0

            if seq_len > 1:
                prev_score = float(np.mean([pattern[i, i - 1] for i in range(1, seq_len)]))

            diag_score = float(np.mean([pattern[i, i] for i in range(seq_len)]))
            bos_score = float(np.mean(pattern[:, 0]))

            # Entropy
            clipped = np.clip(pattern, 1e-10, 1.0)
            ent = float(np.mean(-np.sum(clipped * np.log(clipped), axis=-1)))
            max_entropy = np.log(seq_len) if seq_len > 1 else 1.0

            # Classify
            if prev_score >= prev_token_threshold:
                categories["previous_token"].append((l, h))
            elif diag_score >= diagonal_threshold:
                categories["self_attention"].append((l, h))
            elif bos_score >= 0.5:
                categories["bos_attention"].append((l, h))
            elif ent > 0.7 * max_entropy:
                categories["broad"].append((l, h))
            else:
                categories["other"].append((l, h))

    return categories


def head_clustering(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    n_clusters: int = 5,
) -> dict[str, np.ndarray]:
    """Cluster attention heads by behavioral similarity.

    Computes a feature vector for each head (entropy, prev-token score,
    diagonal score, max attention) and clusters them.

    Uses a simple k-means-like approach without external dependencies.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        n_clusters: Number of clusters.

    Returns:
        Dict with:
        - "labels": [n_layers * n_heads] cluster assignment per head
        - "features": [n_layers * n_heads, 4] feature matrix
        - "feature_names": list of feature names
    """
    scores = head_importance_scores(model, tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total = n_layers * n_heads

    # Build feature matrix
    features = np.zeros((total, 4))
    for l in range(n_layers):
        for h in range(n_heads):
            idx = l * n_heads + h
            features[idx, 0] = scores["entropy"][l, h]
            features[idx, 1] = scores["max_attn"][l, h]
            features[idx, 2] = scores["prev_token"][l, h]
            features[idx, 3] = scores["diag_score"][l, h]

    # Normalize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-10
    normed = (features - mean) / std

    # Simple k-means
    rng = np.random.RandomState(42)
    centers = normed[rng.choice(total, n_clusters, replace=False)]

    for _ in range(50):
        # Assign
        dists = np.sum((normed[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = np.mean(normed[mask], axis=0)
            else:
                new_centers[k] = centers[k]

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return {
        "labels": labels,
        "features": features,
        "feature_names": ["entropy", "max_attn", "prev_token", "diag_score"],
    }
