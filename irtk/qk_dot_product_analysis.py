"""QK dot product analysis: pre-softmax attention score structure."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def qk_score_statistics(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0) -> dict:
    """Statistics of pre-softmax QK dot products per head."""
    _, cache = model.run_with_cache(tokens)
    scores = cache[("attn_scores", layer)]  # [n_heads, seq_q, seq_k]

    per_head = []
    for head in range(model.cfg.n_heads):
        s = scores[head]
        # Only look at valid (causal) scores
        mask = jnp.tril(jnp.ones_like(s))
        valid = s * mask + jnp.where(mask, 0, jnp.nan)
        valid_flat = s[mask > 0]

        per_head.append({
            "head": int(head),
            "mean_score": float(jnp.mean(valid_flat)),
            "std_score": float(jnp.std(valid_flat)),
            "max_score": float(jnp.max(valid_flat)),
            "min_score": float(jnp.min(valid_flat)),
        })
    return {
        "layer": layer,
        "per_head": per_head,
    }


def qk_temperature_analysis(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0) -> dict:
    """Effective temperature of attention: how sharp are the pre-softmax scores?

    High temperature (large score variance) -> sharp attention patterns.
    """
    _, cache = model.run_with_cache(tokens)
    scores = cache[("attn_scores", layer)]

    per_head = []
    for head in range(model.cfg.n_heads):
        s = scores[head]
        # Score range per query position
        ranges = []
        for q in range(len(tokens)):
            valid = s[q, :q + 1]
            ranges.append(float(jnp.max(valid) - jnp.min(valid)))
        mean_range = sum(ranges) / len(ranges)
        per_head.append({
            "head": int(head),
            "mean_score_range": mean_range,
            "is_sharp": mean_range > 2.0,
        })

    return {
        "layer": layer,
        "per_head": per_head,
        "n_sharp": sum(1 for h in per_head if h["is_sharp"]),
    }


def qk_positional_bias(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0, head: int = 0) -> dict:
    """How much do QK scores depend on relative position vs content?"""
    _, cache = model.run_with_cache(tokens)
    scores = cache[("attn_scores", layer)][head]  # [seq_q, seq_k]

    # Average score by relative distance
    seq_len = len(tokens)
    distance_scores = {}
    for q in range(seq_len):
        for k in range(q + 1):
            dist = q - k
            if dist not in distance_scores:
                distance_scores[dist] = []
            distance_scores[dist].append(float(scores[q, k]))

    per_distance = []
    for dist in sorted(distance_scores.keys()):
        vals = distance_scores[dist]
        per_distance.append({
            "distance": dist,
            "mean_score": sum(vals) / len(vals),
            "n_samples": len(vals),
        })

    return {
        "layer": layer,
        "head": head,
        "per_distance": per_distance,
    }


def qk_content_vs_position(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Estimate content vs positional contribution to QK scores."""
    _, cache = model.run_with_cache(tokens)
    scores = cache[("attn_scores", layer)]

    per_head = []
    for head in range(model.cfg.n_heads):
        s = scores[head]  # [seq_q, seq_k]
        # Variance explained by position: score correlation with distance
        total_var = 0.0
        position_var = 0.0
        for q in range(len(tokens)):
            valid = s[q, :q + 1]
            total_var += float(jnp.var(valid))
            # Mean score at each relative distance
            mean_by_dist = jnp.array([float(s[q, k]) for k in range(q + 1)])
            position_var += float(jnp.var(mean_by_dist))

        pos_fraction = position_var / max(total_var, 1e-8)
        per_head.append({
            "head": int(head),
            "position_fraction": pos_fraction,
            "content_fraction": 1.0 - pos_fraction,
            "is_positional": pos_fraction > 0.5,
        })

    return {
        "layer": layer,
        "per_head": per_head,
        "n_positional": sum(1 for h in per_head if h["is_positional"]),
    }


def qk_dot_product_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer QK dot product summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        temp = qk_temperature_analysis(model, tokens, layer)
        cp = qk_content_vs_position(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "n_sharp": temp["n_sharp"],
            "n_positional": cp["n_positional"],
        })
    return {"per_layer": per_layer}
