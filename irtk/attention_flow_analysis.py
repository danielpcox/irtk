"""Attention flow analysis: trace information flow through attention.

Analyze how information propagates through multi-layer attention,
including attention rollout, flow matrices, and source attribution.
"""

import jax.numpy as jnp


def attention_rollout(model, tokens, start_layer=0, end_layer=None):
    """Compute attention rollout: accumulated attention across layers.

    Multiplies attention patterns layer by layer with residual connections.

    Returns:
        dict with 'rollout_matrix' [seq, seq], 'per_layer_patterns'.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    if end_layer is None:
        end_layer = n_layers
    seq_len = cache[("pattern", start_layer)].shape[1]
    rollout = jnp.eye(seq_len)
    per_layer = []
    for layer in range(start_layer, end_layer):
        patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
        mean_attn = jnp.mean(patterns, axis=0)  # [seq, seq]
        # Add residual connection
        attn_with_resid = 0.5 * mean_attn + 0.5 * jnp.eye(seq_len)
        # Normalize rows
        row_sums = jnp.sum(attn_with_resid, axis=-1, keepdims=True) + 1e-10
        attn_norm = attn_with_resid / row_sums
        rollout = rollout @ attn_norm
        per_layer.append({
            "layer": layer,
            "pattern": mean_attn,
        })
    return {
        "rollout_matrix": rollout,
        "per_layer_patterns": per_layer,
    }


def source_token_attribution(model, tokens, target_position=-1):
    """How much does each source position contribute to the target?

    Uses attention rollout to attribute influence to source positions.

    Returns:
        dict with 'per_position' list of attributions.
    """
    rollout = attention_rollout(model, tokens)
    matrix = rollout["rollout_matrix"]
    attributions = matrix[target_position]
    seq_len = len(attributions)
    per_position = []
    for pos in range(seq_len):
        per_position.append({
            "position": pos,
            "attribution": float(attributions[pos]),
        })
    sorted_by_attr = sorted(per_position, key=lambda x: -x["attribution"])
    return {
        "per_position": per_position,
        "most_influential": sorted_by_attr[0]["position"],
    }


def layer_attention_entropy(model, tokens):
    """Average attention entropy per layer across heads.

    Returns:
        dict with 'per_layer' list of entropy values.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        patterns = cache[("pattern", layer)]
        n_heads = patterns.shape[0]
        seq_len = patterns.shape[1]
        head_entropies = []
        for h in range(n_heads):
            pat = patterns[h]
            entropy = float(-jnp.sum(pat * jnp.log(pat + 1e-10)) / seq_len)
            head_entropies.append(entropy)
        per_layer.append({
            "layer": layer,
            "mean_entropy": sum(head_entropies) / len(head_entropies),
            "min_entropy": min(head_entropies),
            "max_entropy": max(head_entropies),
        })
    return {"per_layer": per_layer}


def attention_distance_profile(model, tokens, layer=0):
    """How far does each head attend? Profile of attention distance.

    Returns:
        dict with 'per_head' list of mean attention distances.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    positions = jnp.arange(seq_len)
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        total_dist = 0.0
        for q in range(seq_len):
            distances = jnp.abs(positions - q).astype(jnp.float32)
            mean_dist = float(jnp.sum(pat[q] * distances))
            total_dist += mean_dist
        avg_dist = total_dist / seq_len
        per_head.append({
            "head": int(h),
            "mean_distance": avg_dist,
        })
    return {"per_head": per_head}


def attention_flow_summary(model, tokens):
    """Summary of attention flow analysis.

    Returns:
        dict with 'per_layer' list.
    """
    entropy = layer_attention_entropy(model, tokens)
    rollout = attention_rollout(model, tokens)
    return {
        "per_layer": entropy["per_layer"],
        "rollout_shape": list(rollout["rollout_matrix"].shape),
    }
