"""Attention score decomposition: decompose pre-softmax attention scores."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def score_magnitude_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile the magnitude of pre-softmax attention scores.

    Large scores → sharp attention. Small scores → uniform attention.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        scores = cache[f'blocks.{layer}.attn.hook_attn_scores']  # [n_heads, seq, seq]
        for head in range(n_heads):
            s = scores[head]
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            masked = s * mask + (1 - mask) * -1e9
            valid_scores = s[mask.astype(bool)]

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_score': float(jnp.mean(valid_scores)),
                'std_score': float(jnp.std(valid_scores)),
                'max_score': float(jnp.max(valid_scores)),
                'min_score': float(jnp.min(valid_scores)),
                'score_range': float(jnp.max(valid_scores) - jnp.min(valid_scores)),
            })

    return {
        'per_head': per_head,
    }


def score_position_decomposition(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int, position: int = -1) -> dict:
    """Decompose scores at a position into Q/K contributions.

    Shows which keys contribute most to the score.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    q = cache[f'blocks.{layer}.attn.hook_q']  # [seq, n_heads, d_head]
    k = cache[f'blocks.{layer}.attn.hook_k']
    d_head = model.cfg.d_head

    q_vec = q[pos, head, :]  # [d_head]
    q_norm = float(jnp.linalg.norm(q_vec))

    per_key = []
    for s in range(pos + 1):
        k_vec = k[s, head, :]
        score = float(jnp.dot(q_vec, k_vec) / jnp.sqrt(d_head))
        k_norm = float(jnp.linalg.norm(k_vec))
        # Score = ||q|| * ||k|| * cos(q,k) / sqrt(d_head)
        cos = float(jnp.dot(q_vec, k_vec) / (q_norm * k_norm + 1e-10))

        per_key.append({
            'key_position': s,
            'key_token': int(tokens[s]),
            'score': score,
            'key_norm': k_norm,
            'cosine': cos,
            'norm_contribution': q_norm * k_norm / (d_head ** 0.5),
            'direction_contribution': cos,
        })

    per_key.sort(key=lambda x: x['score'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'query_position': pos,
        'query_norm': q_norm,
        'per_key': per_key,
    }


def score_temperature_analysis(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """How sensitive are attention scores to temperature scaling?

    Effective temperature = std of pre-softmax scores.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    scores = cache[f'blocks.{layer}.attn.hook_attn_scores'][head]  # [seq, seq]
    patterns = cache[f'blocks.{layer}.attn.hook_pattern'][head]  # [seq, seq]

    per_position = []
    for pos in range(seq_len):
        row = scores[pos, :pos + 1]
        prob_row = patterns[pos, :pos + 1]

        eff_temp = float(jnp.std(row))
        max_prob = float(jnp.max(prob_row))
        entropy = -float(jnp.sum(prob_row * jnp.log(prob_row + 1e-10)))

        per_position.append({
            'position': pos,
            'effective_temperature': eff_temp,
            'max_probability': max_prob,
            'entropy': entropy,
        })

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
    }


def score_cross_head_comparison(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Compare score distributions across heads at a given position.

    Shows which heads have similar or different score patterns.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    scores = cache[f'blocks.{layer}.attn.hook_attn_scores']  # [n_heads, seq, seq]

    head_scores = []
    for h in range(n_heads):
        row = scores[h, pos, :pos + 1]
        row_normed = row / (jnp.linalg.norm(row) + 1e-10)
        head_scores.append(row_normed)

    head_scores = jnp.stack(head_scores)
    sim_matrix = head_scores @ head_scores.T

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                'head_a': i,
                'head_b': j,
                'score_similarity': float(sim_matrix[i, j]),
            })

    return {
        'layer': layer,
        'position': pos,
        'pairs': pairs,
    }


def score_softmax_saturation(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How saturated is the softmax across heads?

    High saturation = near-one-hot attention. Low = near-uniform.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        scores = cache[f'blocks.{layer}.attn.hook_attn_scores']
        for head in range(n_heads):
            p = patterns[head]
            s = scores[head]

            max_probs = []
            score_ranges = []
            for pos in range(seq_len):
                max_probs.append(float(jnp.max(p[pos, :pos + 1])))
                row = s[pos, :pos + 1]
                score_ranges.append(float(jnp.max(row) - jnp.min(row)))

            mean_max = sum(max_probs) / len(max_probs)
            mean_range = sum(score_ranges) / len(score_ranges)

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_max_prob': mean_max,
                'mean_score_range': mean_range,
                'is_saturated': bool(mean_max > 0.8),
            })

    return {
        'per_head': per_head,
        'n_saturated': sum(1 for h in per_head if h['is_saturated']),
    }
