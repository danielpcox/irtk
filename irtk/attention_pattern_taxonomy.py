"""Attention pattern taxonomy: classify and catalog attention patterns.

Automatically identify and categorize attention patterns: diagonal,
uniform, sparse, local, global, stripe, and mixed patterns.
"""

import jax
import jax.numpy as jnp


def pattern_diagonal_score(pattern):
    """Score how diagonal (previous-token-like) a pattern is.

    Args:
        pattern: [seq, seq] attention pattern matrix.

    Returns:
        float: diagonal score (0=not diagonal, 1=perfectly diagonal).
    """
    seq_len = pattern.shape[0]
    if seq_len < 2:
        return 0.0
    score = 0.0
    count = 0
    for i in range(1, seq_len):
        score += float(pattern[i, i - 1])
        count += 1
    return score / count if count > 0 else 0.0


def pattern_uniformity_score(pattern):
    """Score how uniform a pattern is (all positions attend equally).

    Returns:
        float: uniformity score (0=not uniform, 1=perfectly uniform).
    """
    seq_len = pattern.shape[0]
    # For causal attention, uniform means 1/k for position k
    expected_uniform = jnp.zeros_like(pattern)
    for i in range(seq_len):
        n_visible = i + 1
        expected_uniform = expected_uniform.at[i, :n_visible].set(1.0 / n_visible)
    diff = jnp.mean(jnp.abs(pattern - expected_uniform))
    return max(0.0, 1.0 - float(diff) * seq_len)


def pattern_sparsity_score(pattern):
    """Score how sparse (concentrated on few positions) a pattern is.

    Returns:
        float: sparsity score (0=uniform, 1=perfectly sparse).
    """
    seq_len = pattern.shape[0]
    # Use Gini coefficient per row
    ginis = []
    for i in range(seq_len):
        row = jnp.sort(pattern[i, :i + 1])
        n = len(row)
        if n <= 1:
            continue
        indices = jnp.arange(1, n + 1, dtype=jnp.float32)
        gini = float(2 * jnp.sum(indices * row) / (n * jnp.sum(row) + 1e-10) - (n + 1) / n)
        ginis.append(gini)
    return sum(ginis) / len(ginis) if ginis else 0.0


def pattern_locality_score(pattern, window=3):
    """Score how local a pattern is (attention concentrated near query).

    Returns:
        float: locality score (0=global, 1=perfectly local).
    """
    seq_len = pattern.shape[0]
    local_mass = 0.0
    count = 0
    for i in range(seq_len):
        start = max(0, i - window)
        local_mass += float(jnp.sum(pattern[i, start:i + 1]))
        count += 1
    return local_mass / count if count > 0 else 0.0


def classify_attention_patterns(model, tokens, layer=0):
    """Classify each head's attention pattern into a category.

    Categories: diagonal, uniform, sparse, local, mixed.

    Returns:
        dict with 'per_head' classifications and scores.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]
        diag = pattern_diagonal_score(pat)
        unif = pattern_uniformity_score(pat)
        sparse = pattern_sparsity_score(pat)
        local = pattern_locality_score(pat)
        scores = {
            "diagonal": diag,
            "uniform": unif,
            "sparse": sparse,
            "local": local,
        }
        category = max(scores, key=scores.get)
        if max(scores.values()) < 0.3:
            category = "mixed"
        per_head.append({
            "head": int(h),
            "category": category,
            "scores": scores,
            "dominant_score": max(scores.values()),
        })
    return {"per_head": per_head}


def attention_pattern_taxonomy_summary(model, tokens):
    """Taxonomy summary across all layers.

    Returns:
        dict with 'per_layer' category counts and 'overall' distribution.
    """
    n_layers = len(model.blocks)
    per_layer = []
    category_counts = {}
    for layer in range(n_layers):
        result = classify_attention_patterns(model, tokens, layer=layer)
        counts = {}
        for h in result["per_head"]:
            cat = h["category"]
            counts[cat] = counts.get(cat, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1
        per_layer.append({
            "layer": layer,
            "category_counts": counts,
        })
    return {
        "per_layer": per_layer,
        "overall_distribution": category_counts,
    }
