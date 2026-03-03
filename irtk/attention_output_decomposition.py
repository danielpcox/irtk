"""Attention output decomposition.

Decompose attention head outputs into interpretable components: per-source
decomposition, value-weighted analysis, output direction alignment to
vocabulary, and per-head logit effect.
"""

import jax
import jax.numpy as jnp


def per_source_output_decomposition(model, tokens, layer, head, position=-1):
    """Decompose a head's output at a position by source token contribution.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index
        position: query position

    Returns:
        dict with per-source decomposition.
    """
    _, cache = model.run_with_cache(tokens)
    pos = position if position >= 0 else len(tokens) - 1
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    attn_weights = pattern[head, pos, :]  # [seq]
    v_h = v[:, head, :]  # [seq, d_head]

    per_source = []
    for src in range(len(tokens)):
        weighted_v = attn_weights[src] * v_h[src]  # [d_head]
        output = weighted_v @ W_O  # [d_model]
        norm = float(jnp.linalg.norm(output))
        per_source.append({
            'source_position': src,
            'source_token': int(tokens[src]),
            'attention_weight': float(attn_weights[src]),
            'output_norm': norm,
        })

    per_source.sort(key=lambda s: -s['output_norm'])
    return {
        'layer': layer,
        'head': head,
        'position': pos,
        'per_source': per_source,
        'top_source': per_source[0] if per_source else None,
    }


def output_vocabulary_alignment(model, tokens, layer, head, position=-1, top_k=5):
    """Find which vocabulary tokens a head's output aligns with.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index
        position: position to analyze
        top_k: number of top tokens

    Returns:
        dict with vocabulary alignment.
    """
    _, cache = model.run_with_cache(tokens)
    pos = position if position >= 0 else len(tokens) - 1
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    z_h = z[pos, head, :]  # [d_head]
    output = z_h @ W_O  # [d_model]
    logits = output @ W_U  # [d_vocab]

    top_indices = jnp.argsort(-logits)[:top_k]
    bottom_indices = jnp.argsort(logits)[:top_k]

    promoted = [{'token': int(idx), 'logit': float(logits[idx])} for idx in top_indices]
    suppressed = [{'token': int(idx), 'logit': float(logits[idx])} for idx in bottom_indices]

    return {
        'layer': layer,
        'head': head,
        'position': pos,
        'promoted_tokens': promoted,
        'suppressed_tokens': suppressed,
        'output_norm': float(jnp.linalg.norm(output)),
    }


def head_logit_effect_profile(model, tokens, layer, head):
    """Profile a head's logit effect across all positions.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index

    Returns:
        dict with per-position logit effects.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    logits = model(tokens)

    per_position = []
    for pos in range(len(tokens)):
        z_h = z[pos, head, :]
        output = z_h @ W_O  # [d_model]
        target = int(jnp.argmax(logits[pos]))
        logit_contrib = float(output @ W_U[:, target])
        output_norm = float(jnp.linalg.norm(output))

        per_position.append({
            'position': pos,
            'target_token': target,
            'logit_contribution': logit_contrib,
            'output_norm': output_norm,
        })

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
        'mean_logit_contribution': float(jnp.mean(jnp.array([p['logit_contribution'] for p in per_position]))),
    }


def attention_weighted_value_analysis(model, tokens, layer, head):
    """Analyze the attention-weighted value mixture.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index

    Returns:
        dict with attention-weighted value analysis.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]

    attn = pattern[head]  # [seq, seq]
    v_h = v[:, head, :]  # [seq, d_head]

    per_position = []
    for pos in range(len(tokens)):
        weights = attn[pos]  # [seq]
        weighted_v = jnp.sum(weights[:, None] * v_h, axis=0)  # [d_head]

        # How concentrated is attention?
        entropy = -float(jnp.sum(weights * jnp.log(weights + 1e-10)))
        max_weight = float(jnp.max(weights))

        # Value diversity: how different are the attended-to values?
        attended_mask = weights > 0.05
        n_attended = int(jnp.sum(attended_mask))

        per_position.append({
            'position': pos,
            'attention_entropy': entropy,
            'max_attention': max_weight,
            'n_attended_sources': n_attended,
            'weighted_value_norm': float(jnp.linalg.norm(weighted_v)),
        })

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
        'mean_entropy': float(jnp.mean(jnp.array([p['attention_entropy'] for p in per_position]))),
    }


def head_output_direction_stability(model, tokens, layer, head):
    """Measure how stable the output direction is across positions.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        head: head index

    Returns:
        dict with direction stability.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    outputs = z[:, head, :] @ W_O  # [seq, d_model]
    norms = jnp.linalg.norm(outputs, axis=-1, keepdims=True) + 1e-10
    normalized = outputs / norms

    # Pairwise cosine similarities
    sim_matrix = normalized @ normalized.T  # [seq, seq]
    n = len(tokens)
    mask = 1.0 - jnp.eye(n)
    mean_similarity = float(jnp.sum(sim_matrix * mask) / (n * (n - 1) + 1e-10))

    # Mean direction
    mean_dir = jnp.mean(normalized, axis=0)
    mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)

    per_position = []
    for pos in range(n):
        alignment = float(jnp.sum(normalized[pos] * mean_dir))
        per_position.append({
            'position': pos,
            'alignment_to_mean': alignment,
            'output_norm': float(norms[pos, 0]),
        })

    return {
        'layer': layer,
        'head': head,
        'mean_pairwise_similarity': mean_similarity,
        'is_stable': mean_similarity > 0.5,
        'per_position': per_position,
    }
