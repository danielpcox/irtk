"""Head QK circuit analysis.

Analyze QK circuits: what each head attends to, positional vs content
patterns, key-query alignment, and attention selectivity.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def qk_alignment_profile(model, tokens, layer=0, pos=-1):
    """Analyze how Q and K vectors align at each position.

    Returns:
        dict with per_head QK alignment info.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    q_key = f'blocks.{layer}.attn.hook_q'
    k_key = f'blocks.{layer}.attn.hook_k'

    if q_key not in cache or k_key not in cache:
        return {'per_head': []}

    q = cache[q_key]  # [seq, n_heads, d_head]
    k = cache[k_key]
    n_heads = q.shape[1]

    results = []
    for h in range(n_heads):
        q_h = q[pos, h]  # [d_head]
        k_all = k[:, h]  # [seq, d_head]

        # Raw dot products (pre-scaling)
        dots = k_all @ q_h  # [seq]

        # Which key positions align most with this query
        top_idx = int(jnp.argmax(dots))
        max_dot = float(dots[top_idx])

        # Selectivity: how peaked is the dot product distribution
        dot_std = float(jnp.std(dots))
        dot_range = float(jnp.max(dots) - jnp.min(dots))

        results.append({
            'head': h,
            'max_key_position': top_idx,
            'max_dot_product': max_dot,
            'dot_product_std': dot_std,
            'dot_product_range': dot_range,
            'q_norm': float(jnp.linalg.norm(q_h)),
        })

    return {'layer': layer, 'query_position': pos, 'per_head': results}


def positional_vs_content_attention(model, tokens, layer=0):
    """Distinguish positional from content-based attention patterns.

    Returns:
        dict with per_head positional vs content metrics.
    """
    cache = _run_and_cache(model, tokens)

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    if pattern_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
    n_heads = pattern.shape[0]
    seq_len = pattern.shape[1]

    results = []
    for h in range(n_heads):
        p = pattern[h]  # [seq_q, seq_k]

        # Mean pattern (averaged across queries -> positional component)
        mean_pattern = jnp.mean(p, axis=0)  # [seq_k]

        # Variance of each key position's attention across queries
        # High variance = content-dependent; Low variance = positional
        per_key_variance = jnp.var(p, axis=0)  # [seq_k]
        mean_variance = float(jnp.mean(per_key_variance))

        # Positional score: how well does mean pattern explain individual rows?
        explained = []
        for q_pos in range(seq_len):
            cos = jnp.dot(p[q_pos], mean_pattern) / jnp.maximum(
                jnp.linalg.norm(p[q_pos]) * jnp.linalg.norm(mean_pattern), 1e-10)
            explained.append(float(cos))
        mean_explained = sum(explained) / len(explained)

        results.append({
            'head': h,
            'positional_score': mean_explained,
            'content_variance': mean_variance,
            'is_positional': mean_explained > 0.8,
        })

    return {'layer': layer, 'per_head': results}


def attention_selectivity(model, tokens, layer=0):
    """How selective is each head's attention?

    Returns:
        dict with per_head selectivity metrics.
    """
    cache = _run_and_cache(model, tokens)

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    if pattern_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]
    n_heads = pattern.shape[0]
    seq_len = pattern.shape[1]

    results = []
    for h in range(n_heads):
        p = pattern[h]  # [seq_q, seq_k]

        # Mean entropy
        p_safe = jnp.maximum(p, 1e-10)
        entropies = -jnp.sum(p * jnp.log(p_safe), axis=-1)  # [seq_q]
        mean_entropy = float(jnp.mean(entropies))
        max_entropy = float(jnp.log(jnp.array(seq_len, dtype=jnp.float32)))

        # Mean max attention weight
        max_weights = jnp.max(p, axis=-1)  # [seq_q]
        mean_max = float(jnp.mean(max_weights))

        # Gini coefficient of attention
        sorted_p = jnp.sort(p, axis=-1)
        n = seq_len
        indices = jnp.arange(1, n + 1)
        gini_per_q = (2.0 * jnp.sum(indices * sorted_p, axis=-1) / jnp.maximum(n * jnp.sum(sorted_p, axis=-1), 1e-10)) - (n + 1.0) / n
        mean_gini = float(jnp.mean(gini_per_q))

        results.append({
            'head': h,
            'mean_entropy': mean_entropy,
            'normalized_entropy': mean_entropy / max(max_entropy, 1e-10),
            'mean_max_weight': mean_max,
            'gini_coefficient': mean_gini,
        })

    return {'layer': layer, 'per_head': results}


def key_query_subspace(model, layer=0, n_components=3):
    """Analyze the key and query subspaces from weights.

    Returns:
        dict with per_head subspace analysis.
    """
    W_Q = model.blocks[layer].attn.W_Q  # [n_heads, d_model, d_head]
    W_K = model.blocks[layer].attn.W_K
    n_heads = W_Q.shape[0]

    results = []
    for h in range(n_heads):
        # QK matrix: W_Q[h]^T @ W_K[h] (what queries match what keys)
        QK = W_Q[h].T @ W_K[h]  # [d_head, d_head]

        S = jnp.linalg.svd(QK, compute_uv=False)
        total = float(jnp.sum(S))
        top_k_var = float(jnp.sum(S[:n_components])) / max(total, 1e-10)

        # Effective rank
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_rank = float(jnp.exp(jnp.array(entropy)))

        results.append({
            'head': h,
            'effective_rank': eff_rank,
            'top_k_concentration': top_k_var,
            'top_singular_value': float(S[0]),
            'condition_number': float(S[0] / jnp.maximum(S[-1], 1e-10)),
        })

    return {'layer': layer, 'per_head': results}


def attention_pattern_type(model, tokens, layer=0):
    """Classify each head's attention pattern type.

    Types: diagonal (self-attend), previous-token, uniform, sparse, other.

    Returns:
        dict with per_head pattern type classification.
    """
    cache = _run_and_cache(model, tokens)

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    if pattern_key not in cache:
        return {'per_head': []}

    pattern = cache[pattern_key]
    n_heads = pattern.shape[0]
    seq_len = pattern.shape[1]

    results = []
    for h in range(n_heads):
        p = pattern[h]  # [seq_q, seq_k]

        # Diagonal score: mean attention on diagonal
        diag_score = float(jnp.mean(jnp.diag(p[:seq_len, :seq_len])))

        # Previous token score: mean attention on position q-1
        prev_scores = []
        for q in range(1, seq_len):
            prev_scores.append(float(p[q, q - 1]))
        prev_score = sum(prev_scores) / max(len(prev_scores), 1) if prev_scores else 0.0

        # Uniform score: how close to 1/seq_len
        uniform_pattern = jnp.ones_like(p) / seq_len
        uniform_cos = float(jnp.sum(p * uniform_pattern) /
                           jnp.maximum(jnp.linalg.norm(p) * jnp.linalg.norm(uniform_pattern), 1e-10))

        # Sparse score: mean max weight
        sparse_score = float(jnp.mean(jnp.max(p, axis=-1)))

        # Classify
        if diag_score > 0.5:
            pattern_type = 'diagonal'
        elif prev_score > 0.5:
            pattern_type = 'previous_token'
        elif uniform_cos > 0.95:
            pattern_type = 'uniform'
        elif sparse_score > 0.7:
            pattern_type = 'sparse'
        else:
            pattern_type = 'mixed'

        results.append({
            'head': h,
            'pattern_type': pattern_type,
            'diagonal_score': diag_score,
            'previous_token_score': prev_score,
            'uniform_score': uniform_cos,
            'sparse_score': sparse_score,
        })

    return {'layer': layer, 'per_head': results}
