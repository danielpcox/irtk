"""Attention pattern dynamics: how patterns change across positions and contexts."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def position_dependent_attention(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Analyze how attention patterns differ at each query position.

    Computes per-position entropy, max attention, and attended source.
    """
    _, cache = model.run_with_cache(tokens)
    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern = cache[pattern_key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    per_position = []
    for pos in range(seq_len):
        row = pattern[pos, :pos+1]
        entropy = float(-jnp.sum(row * jnp.log(row + 1e-10)))
        max_attn = float(jnp.max(row))
        max_source = int(jnp.argmax(row))
        n_attended = int(jnp.sum(row > 1.0 / (pos + 1)))

        per_position.append({
            'position': pos,
            'entropy': entropy,
            'max_attention': max_attn,
            'max_source': max_source,
            'n_attended_sources': n_attended,
            'attends_self': max_source == pos,
        })

    # Entropy trend
    entropies = [p['entropy'] for p in per_position]
    if len(entropies) >= 2:
        entropy_trend = (entropies[-1] - entropies[0]) / (len(entropies) - 1)
    else:
        entropy_trend = 0.0

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
        'entropy_trend': entropy_trend,
        'mean_entropy': sum(entropies) / len(entropies),
    }


def attention_shift_between_contexts(model: HookedTransformer, tokens_a: jnp.ndarray, tokens_b: jnp.ndarray, layer: int, head: int) -> dict:
    """How does a head's attention pattern change between two inputs?

    Compares pattern structure: entropy, max attention, similarity.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern_a = cache_a[pattern_key][head]
    pattern_b = cache_b[pattern_key][head]

    min_len = min(pattern_a.shape[0], pattern_b.shape[0])

    per_position = []
    for pos in range(min_len):
        row_a = pattern_a[pos, :pos+1]
        row_b = pattern_b[pos, :pos+1]

        entropy_a = float(-jnp.sum(row_a * jnp.log(row_a + 1e-10)))
        entropy_b = float(-jnp.sum(row_b * jnp.log(row_b + 1e-10)))

        # JS divergence as similarity
        m = 0.5 * (row_a + row_b)
        kl_am = jnp.sum(row_a * jnp.log(row_a / (m + 1e-10) + 1e-10))
        kl_bm = jnp.sum(row_b * jnp.log(row_b / (m + 1e-10) + 1e-10))
        js_div = float(0.5 * (kl_am + kl_bm))

        per_position.append({
            'position': pos,
            'entropy_a': entropy_a,
            'entropy_b': entropy_b,
            'entropy_change': entropy_b - entropy_a,
            'js_divergence': js_div,
        })

    mean_js = sum(p['js_divergence'] for p in per_position) / len(per_position) if per_position else 0.0

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
        'mean_js_divergence': mean_js,
        'is_context_sensitive': mean_js > 0.1,
    }


def attention_entropy_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile entropy across all heads at all layers.

    Provides a map of which heads are sharp vs diffuse.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head = []
    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        patterns = cache[pattern_key]

        for head in range(n_heads):
            pattern = patterns[head]
            entropies = -jnp.sum(pattern * jnp.log(pattern + 1e-10), axis=-1)
            mean_ent = float(jnp.mean(entropies))
            min_ent = float(jnp.min(entropies))
            max_ent = float(jnp.max(entropies))

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_entropy': mean_ent,
                'min_entropy': min_ent,
                'max_entropy': max_ent,
                'is_sharp': mean_ent < 1.0,
            })

    n_sharp = sum(1 for h in per_head if h['is_sharp'])

    return {
        'per_head': per_head,
        'n_sharp_heads': n_sharp,
        'n_diffuse_heads': len(per_head) - n_sharp,
        'mean_entropy': sum(h['mean_entropy'] for h in per_head) / len(per_head),
    }


def attention_distance_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """What distances does this head attend to?

    Measures the distribution of attention over source-target distances.
    """
    _, cache = model.run_with_cache(tokens)
    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern = cache[pattern_key][head]
    seq_len = pattern.shape[0]

    # Per-position mean attended distance
    per_position = []
    distance_weights = {}

    for pos in range(seq_len):
        row = pattern[pos, :pos+1]
        sources = jnp.arange(pos + 1, dtype=jnp.float32)
        distances = pos - sources

        mean_dist = float(jnp.sum(row * distances))
        max_dist = pos

        per_position.append({
            'position': pos,
            'mean_distance': mean_dist,
            'max_possible_distance': max_dist,
            'relative_distance': mean_dist / (max_dist + 1e-10),
        })

        # Accumulate distance distribution
        for d in range(pos + 1):
            dist = pos - d
            w = float(row[d])
            distance_weights[dist] = distance_weights.get(dist, 0.0) + w

    mean_distance = sum(p['mean_distance'] for p in per_position) / len(per_position) if per_position else 0.0
    is_local = mean_distance < 2.0

    return {
        'layer': layer,
        'head': head,
        'per_position': per_position,
        'mean_attention_distance': mean_distance,
        'is_local': is_local,
    }


def attention_pattern_rank(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Compute the effective rank of the attention pattern matrix.

    Low rank = stereotyped pattern; high rank = diverse attention.
    """
    _, cache = model.run_with_cache(tokens)
    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern = cache[pattern_key][head]
    seq_len = pattern.shape[0]

    # SVD of the lower-triangular attention pattern
    svd_vals = jnp.linalg.svd(pattern, compute_uv=False)
    svd_normalized = svd_vals / (jnp.sum(svd_vals) + 1e-10)
    effective_rank = float(jnp.exp(-jnp.sum(svd_normalized * jnp.log(svd_normalized + 1e-10))))

    # Variance explained
    variance = svd_vals ** 2
    total_var = jnp.sum(variance) + 1e-10
    cumulative = jnp.cumsum(variance) / total_var
    rank_90 = int(jnp.searchsorted(cumulative, 0.9) + 1)

    return {
        'layer': layer,
        'head': head,
        'effective_rank': effective_rank,
        'max_rank': seq_len,
        'rank_utilization': effective_rank / seq_len,
        'rank_90_pct': rank_90,
        'top_singular_value': float(svd_vals[0]),
        'is_low_rank': effective_rank < seq_len * 0.3,
    }
