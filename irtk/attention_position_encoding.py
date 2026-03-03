"""Attention position encoding: how position information is encoded and used."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def positional_attention_bias(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0) -> dict:
    """How much attention patterns depend on relative position vs content.

    Compares attention scores to what a purely positional pattern would look like.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = pattern.shape[1]

    # Expected positional pattern: uniform over causal positions
    causal = jnp.tril(jnp.ones((seq_len, seq_len)))
    pos_pattern = causal / jnp.sum(causal, axis=-1, keepdims=True)

    per_head = []
    for head in range(model.cfg.n_heads):
        p = pattern[head]  # [seq, seq]
        # Correlation with positional pattern
        p_flat = p.flatten()
        pp_flat = pos_pattern.flatten()
        corr = float(jnp.corrcoef(jnp.stack([p_flat, pp_flat]))[0, 1])
        per_head.append({
            "head": int(head),
            "positional_correlation": corr if not jnp.isnan(corr) else 0.0,
            "is_positional": abs(corr if not jnp.isnan(corr) else 0.0) > 0.5,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "n_positional_heads": sum(1 for h in per_head if h["is_positional"]),
    }


def position_sensitivity(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0) -> dict:
    """How sensitive each head's attention is to the query position.

    Compares patterns from different query positions.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = pattern.shape[1]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = pattern[head]  # [seq, seq]
        # Pairwise cosine similarity of attention distributions from different positions
        if seq_len > 1:
            norms = jnp.sqrt(jnp.sum(p ** 2, axis=-1, keepdims=True)).clip(1e-8)
            normed = p / norms
            sim = normed @ normed.T
            mask = 1.0 - jnp.eye(seq_len)
            mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))
        else:
            mean_sim = 1.0

        per_head.append({
            "head": int(head),
            "pattern_similarity": mean_sim,
            "is_position_sensitive": mean_sim < 0.5,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_sensitivity": sum(1 for h in per_head if h["is_position_sensitive"]) / len(per_head),
    }


def relative_position_preference(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0, head: int = 0) -> dict:
    """Attention mass as a function of relative position (query - key).

    Shows if the head prefers recent tokens, distant tokens, etc.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]
    p = pattern[head]  # [seq, seq]
    seq_len = p.shape[0]

    # Accumulate attention by relative position
    max_dist = seq_len - 1
    rel_mass = {}
    rel_count = {}
    for q in range(seq_len):
        for k in range(q + 1):
            dist = q - k
            if dist not in rel_mass:
                rel_mass[dist] = 0.0
                rel_count[dist] = 0
            rel_mass[dist] += float(p[q, k])
            rel_count[dist] += 1

    per_distance = []
    for dist in sorted(rel_mass.keys()):
        per_distance.append({
            "distance": dist,
            "mean_attention": rel_mass[dist] / max(rel_count[dist], 1),
            "total_mass": rel_mass[dist],
        })

    # Find peak distance
    if per_distance:
        peak = max(per_distance, key=lambda d: d["mean_attention"])
        peak_distance = peak["distance"]
    else:
        peak_distance = 0

    return {
        "layer": layer,
        "head": head,
        "per_distance": per_distance,
        "peak_distance": peak_distance,
        "prefers_recent": peak_distance <= 1,
    }


def position_encoding_strength(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How strongly position is encoded in the residual stream at each layer.

    Compares similarity of representations at different positions.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]  # [seq, d_model]
        seq_len = resid.shape[0]
        norms = jnp.sqrt(jnp.sum(resid ** 2, axis=-1, keepdims=True)).clip(1e-8)
        normed = resid / norms
        sim = normed @ normed.T
        mask = 1.0 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

        per_layer.append({
            "layer": layer,
            "mean_position_similarity": mean_sim,
            "position_distinct": mean_sim < 0.5,
        })
    return {
        "per_layer": per_layer,
        "most_distinct_layer": min(range(len(per_layer)),
                                     key=lambda i: per_layer[i]["mean_position_similarity"]),
    }


def position_encoding_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Combined position encoding analysis."""
    per_layer = []
    strength = position_encoding_strength(model, tokens)
    for layer in range(model.cfg.n_layers):
        bias = positional_attention_bias(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "n_positional_heads": bias["n_positional_heads"],
            "position_similarity": strength["per_layer"][layer]["mean_position_similarity"],
        })
    return {"per_layer": per_layer}
