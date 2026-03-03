"""Attention sink analysis: detect and characterize attention sinks.

Identify positions that receive disproportionate attention (sinks),
analyze their formation across layers, and measure their effect.
"""

import jax.numpy as jnp


def attention_sink_detection(model, tokens, layer=0, threshold=0.3):
    """Find positions that receive disproportionate attention.

    A sink position receives more than threshold of total attention
    across most query positions.

    Returns:
        dict with 'per_head' list, 'n_sinks' count.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    total_sinks = 0
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        received = jnp.mean(pat, axis=0)  # mean attention received per key position
        sinks = []
        for pos in range(seq_len):
            if float(received[pos]) > threshold:
                sinks.append({
                    "position": int(pos),
                    "mean_attention_received": float(received[pos]),
                })
        per_head.append({
            "head": int(h),
            "sinks": sinks,
            "n_sinks": len(sinks),
        })
        total_sinks += len(sinks)
    return {"per_head": per_head, "n_sinks": total_sinks}


def bos_attention_profile(model, tokens, position=0):
    """How much attention does the BOS/first position receive across all heads?

    Returns:
        dict with 'per_layer' list of per-head BOS attention scores.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        patterns = cache[("pattern", layer)]
        n_heads = patterns.shape[0]
        per_head = []
        for h in range(n_heads):
            bos_attn = float(jnp.mean(patterns[h, :, position]))
            per_head.append({
                "head": int(h),
                "bos_attention": bos_attn,
            })
        mean_bos = sum(p["bos_attention"] for p in per_head) / len(per_head)
        per_layer.append({
            "layer": layer,
            "per_head": per_head,
            "mean_bos_attention": mean_bos,
        })
    return {"per_layer": per_layer}


def sink_formation_trajectory(model, tokens, position=0):
    """Track how a sink position's received attention evolves across layers.

    Returns:
        dict with 'per_layer' list tracking the sink strength at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        patterns = cache[("pattern", layer)]
        n_heads = patterns.shape[0]
        head_attns = []
        for h in range(n_heads):
            head_attns.append(float(jnp.mean(patterns[h, :, position])))
        mean_attn = sum(head_attns) / len(head_attns)
        max_attn = max(head_attns)
        per_layer.append({
            "layer": layer,
            "mean_attention": mean_attn,
            "max_attention": max_attn,
        })
    return {"per_layer": per_layer, "position": position}


def attention_concentration(model, tokens, layer=0):
    """How concentrated is the attention distribution per head?

    Measures the Gini coefficient of attention patterns.

    Returns:
        dict with 'per_head' list with gini coefficient, 'mean_gini'.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        flat = pat.reshape(-1)
        sorted_vals = jnp.sort(flat)
        n = len(sorted_vals)
        index = jnp.arange(1, n + 1)
        gini = float((2.0 * jnp.sum(index * sorted_vals) / (n * jnp.sum(sorted_vals) + 1e-10)) - (n + 1) / n)
        per_head.append({
            "head": int(h),
            "gini": gini,
        })
    mean_gini = sum(p["gini"] for p in per_head) / len(per_head)
    return {"per_head": per_head, "mean_gini": mean_gini}


def attention_sink_summary(model, tokens):
    """Summary of attention sink analysis across all layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        sinks = attention_sink_detection(model, tokens, layer=layer)
        conc = attention_concentration(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "n_sinks": sinks["n_sinks"],
            "mean_gini": conc["mean_gini"],
        })
    return {"per_layer": per_layer}
