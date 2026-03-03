"""Attention-value composition analysis.

Analyze how attention patterns and value vectors combine to produce
head outputs, including pattern-value alignment, composition effects,
and source-specific contributions.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def pattern_value_alignment(model, tokens, layer=0, pos=-1):
    """How well do attention patterns match value magnitude?

    High alignment: head attends more to positions with larger values.
    Low alignment: pattern and value importance are decorrelated.

    Returns:
        dict with per_head list containing alignment scores.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    v_key = f'blocks.{layer}.attn.hook_v'

    if pattern_key not in cache or v_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
    v = cache[v_key]  # [seq, n_heads, d_head]
    n_heads = pattern.shape[0]

    results = []
    for h in range(n_heads):
        attn = pattern[h, pos, :seq_len]  # [seq_k]
        v_norms = jnp.linalg.norm(v[:, h, :], axis=-1)  # [seq]

        # Correlation between attention and value norms
        attn_centered = attn - jnp.mean(attn)
        v_centered = v_norms - jnp.mean(v_norms)
        corr = float(jnp.dot(attn_centered, v_centered) /
                      jnp.maximum(jnp.linalg.norm(attn_centered) * jnp.linalg.norm(v_centered), 1e-10))

        # Effective contribution: attention-weighted value norm
        effective_contrib = float(jnp.sum(attn * v_norms))

        results.append({
            'head': h,
            'pattern_value_correlation': corr,
            'effective_contribution': effective_contrib,
            'max_attention': float(jnp.max(attn)),
            'max_value_norm': float(jnp.max(v_norms)),
        })

    return {'layer': layer, 'per_head': results}


def source_value_decomposition(model, tokens, layer=0, pos=-1, top_k=3):
    """Decompose head output by source position contributions.

    Returns:
        dict with per_head list, each containing source breakdown.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    v_key = f'blocks.{layer}.attn.hook_v'

    if pattern_key not in cache or v_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]
    v = cache[v_key]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    n_heads = pattern.shape[0]

    results = []
    for h in range(n_heads):
        attn = pattern[h, pos, :seq_len]

        contributions = []
        for s in range(seq_len):
            # This source's contribution to head output in d_model space
            weighted_v = attn[s] * v[s, h]  # [d_head]
            output = jnp.einsum('h,hm->m', weighted_v, W_O[h])  # [d_model]
            norm = float(jnp.linalg.norm(output))
            contributions.append({
                'source': s,
                'attention_weight': float(attn[s]),
                'output_norm': norm,
            })

        contributions.sort(key=lambda x: -x['output_norm'])

        # Total output norm
        total_output = jnp.einsum('h,hm->m',
                                  jnp.einsum('s,sh->h', attn[:seq_len], v[:seq_len, h]),
                                  W_O[h])
        total_norm = float(jnp.linalg.norm(total_output))

        results.append({
            'head': h,
            'top_sources': contributions[:top_k],
            'total_output_norm': total_norm,
            'top_source_fraction': contributions[0]['output_norm'] / max(total_norm, 1e-10) if contributions else 0,
        })

    return {'layer': layer, 'per_head': results}


def value_mixing_analysis(model, tokens, layer=0, pos=-1):
    """Analyze how values from different positions mix through attention.

    Returns:
        dict with per_head mixing entropy and diversity metrics.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    v_key = f'blocks.{layer}.attn.hook_v'

    if pattern_key not in cache or v_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]
    v = cache[v_key]
    n_heads = pattern.shape[0]

    results = []
    for h in range(n_heads):
        attn = pattern[h, pos, :seq_len]

        # Value diversity: how different are the attended values?
        v_h = v[:seq_len, h]  # [seq, d_head]
        # Pairwise cosine similarity of values weighted by attention
        v_norms = jnp.linalg.norm(v_h, axis=-1, keepdims=True)
        v_normalized = v_h / jnp.maximum(v_norms, 1e-10)
        pairwise_cos = v_normalized @ v_normalized.T  # [seq, seq]

        # Attention-weighted diversity
        weighted_sim = float(jnp.sum(jnp.outer(attn, attn) * pairwise_cos))

        # Mixing entropy from attention distribution
        attn_safe = jnp.maximum(attn, 1e-10)
        mixing_entropy = -float(jnp.sum(attn * jnp.log(attn_safe)))

        results.append({
            'head': h,
            'mixing_entropy': mixing_entropy,
            'weighted_similarity': weighted_sim,
            'value_diversity': 1.0 - weighted_sim,
            'n_effective_sources': float(jnp.exp(jnp.array(mixing_entropy))),
        })

    return {'layer': layer, 'per_head': results}


def composition_with_previous_layer(model, tokens, layer=1, pos=-1):
    """Analyze how this layer's attention composes with the previous layer.

    Q-composition: does this layer's Q attend based on previous layer's output?
    V-composition: does this layer read values written by previous layer?

    Returns:
        dict with per_head composition scores.
    """
    if layer == 0:
        return {'per_head': [], 'note': 'Layer 0 has no previous layer'}

    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    # Previous layer's head outputs
    prev_z_key = f'blocks.{layer - 1}.attn.hook_z'
    prev_pattern_key = f'blocks.{layer - 1}.attn.hook_pattern'
    curr_pattern_key = f'blocks.{layer}.attn.hook_pattern'
    curr_q_key = f'blocks.{layer}.attn.hook_q'
    curr_v_key = f'blocks.{layer}.attn.hook_v'

    if curr_pattern_key not in cache or prev_z_key not in cache:
        return {'per_head': []}

    curr_pattern = cache[curr_pattern_key]
    prev_z = cache[prev_z_key]  # [seq, n_heads, d_head]
    W_O_prev = model.blocks[layer - 1].attn.W_O

    n_heads_curr = curr_pattern.shape[0]
    n_heads_prev = prev_z.shape[1]

    results = []
    for h_curr in range(n_heads_curr):
        max_v_comp = 0.0
        best_prev_head = 0

        for h_prev in range(n_heads_prev):
            # Previous head's output at each position
            prev_output = jnp.einsum('sh,hm->sm', prev_z[:, h_prev, :], W_O_prev[h_prev])  # [seq, d_model]

            # Current head's value at pos comes from residual which includes prev output
            # V-composition: correlation of attention-weighted prev outputs with current values
            if curr_v_key in cache:
                curr_v = cache[curr_v_key][:, h_curr, :]  # [seq, d_head]
                prev_norms = jnp.linalg.norm(prev_output, axis=-1)  # [seq]
                curr_v_norms = jnp.linalg.norm(curr_v, axis=-1)  # [seq]
                v_comp = float(jnp.abs(jnp.corrcoef(prev_norms, curr_v_norms)[0, 1]))
                if jnp.isnan(jnp.array(v_comp)):
                    v_comp = 0.0
                if v_comp > max_v_comp:
                    max_v_comp = v_comp
                    best_prev_head = h_prev

        results.append({
            'head': h_curr,
            'max_v_composition': max_v_comp,
            'best_prev_head': best_prev_head,
        })

    return {'layer': layer, 'per_head': results}


def attention_output_logit_decomposition(model, tokens, layer=0, pos=-1, top_k=5):
    """Decompose each head's logit contribution by source position.

    Returns:
        dict with per_head list of source-specific logit contributions.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    v_key = f'blocks.{layer}.attn.hook_v'

    if pattern_key not in cache or v_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]
    v = cache[v_key]
    W_O = model.blocks[layer].attn.W_O
    W_U = model.unembed.W_U
    n_heads = pattern.shape[0]

    # Get target token
    logits = model(tokens)
    target = int(jnp.argmax(logits[pos]))
    W_U_col = W_U[:, target]

    results = []
    for h in range(n_heads):
        attn = pattern[h, pos, :seq_len]

        source_logits = []
        for s in range(seq_len):
            weighted_v = attn[s] * v[s, h]  # [d_head]
            output = jnp.einsum('h,hm->m', weighted_v, W_O[h])  # [d_model]
            logit_contrib = float(jnp.dot(output, W_U_col))
            source_logits.append({
                'source': s,
                'logit_contribution': logit_contrib,
            })

        source_logits.sort(key=lambda x: -abs(x['logit_contribution']))

        total_logit = sum(sl['logit_contribution'] for sl in source_logits)
        results.append({
            'head': h,
            'total_logit_contribution': total_logit,
            'top_sources': source_logits[:top_k],
        })

    return {'layer': layer, 'target_token': target, 'per_head': results}
