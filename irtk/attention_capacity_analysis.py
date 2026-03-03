"""Attention capacity analysis.

Analyze attention head capacity: information throughput, pattern rank,
bottleneck detection, head utilization, and capacity allocation.
"""

import jax
import jax.numpy as jnp


def attention_pattern_rank(model, tokens, layer):
    """Analyze the effective rank of attention patterns per head.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with per-head attention pattern rank analysis.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]

    results = []
    for h in range(n_heads):
        P = patterns[h]  # [seq, seq]
        S = jnp.linalg.svd(P, compute_uv=False)
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_rank = float(jnp.exp(jnp.array(entropy)))
        results.append({
            'head': h,
            'effective_rank': eff_rank,
            'top_singular_value': float(S[0]),
            'rank_ratio': eff_rank / P.shape[0],
            'n_significant': int(jnp.sum(S > 0.01 * S[0])),
        })

    return {
        'layer': layer,
        'per_head': results,
        'mean_effective_rank': float(jnp.mean(jnp.array([r['effective_rank'] for r in results]))),
    }


def attention_information_throughput(model, tokens, layer):
    """Estimate information throughput of each head.

    Measures how much the value information is preserved through attention.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with per-head information throughput estimates.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    n_heads = patterns.shape[0]

    results = []
    for h in range(n_heads):
        P = patterns[h]  # [seq, seq]
        V = v[:, h, :]  # [seq, d_head]

        # Output = P @ V: how much of V's information survives?
        output = P @ V  # [seq, d_head]
        # Output variance relative to input variance
        v_var = float(jnp.var(V))
        o_var = float(jnp.var(output))
        throughput = o_var / max(v_var, 1e-10)

        # Attention entropy (average across query positions)
        P_safe = jnp.maximum(P, 1e-10)
        entropies = -jnp.sum(P * jnp.log(P_safe), axis=-1)
        mean_entropy = float(jnp.mean(entropies))

        results.append({
            'head': h,
            'throughput': throughput,
            'mean_entropy': mean_entropy,
            'output_variance': o_var,
            'input_variance': v_var,
        })

    return {
        'layer': layer,
        'per_head': results,
        'mean_throughput': float(jnp.mean(jnp.array([r['throughput'] for r in results]))),
    }


def attention_bottleneck_detection(model, tokens):
    """Detect attention bottlenecks across layers.

    A bottleneck is a layer where attention heads have low effective rank,
    limiting information flow.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with per-layer bottleneck scores.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    layer_scores = []
    for l in range(n_layers):
        patterns = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]
        n_heads = patterns.shape[0]

        head_ranks = []
        for h in range(n_heads):
            P = patterns[h]
            S = jnp.linalg.svd(P, compute_uv=False)
            S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
            S_safe = jnp.maximum(S_norm, 1e-10)
            entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
            head_ranks.append(float(jnp.exp(jnp.array(entropy))))

        avg_rank = float(jnp.mean(jnp.array(head_ranks)))
        min_rank = float(jnp.min(jnp.array(head_ranks)))
        seq_len = patterns.shape[1]

        layer_scores.append({
            'layer': l,
            'mean_effective_rank': avg_rank,
            'min_effective_rank': min_rank,
            'bottleneck_score': 1.0 - avg_rank / max(seq_len, 1),
            'per_head_ranks': head_ranks,
        })

    # Most bottlenecked layer
    scores = [s['bottleneck_score'] for s in layer_scores]
    most_bottlenecked = int(jnp.argmax(jnp.array(scores)))

    return {
        'per_layer': layer_scores,
        'most_bottlenecked_layer': most_bottlenecked,
        'max_bottleneck_score': float(max(scores)),
    }


def head_utilization(model, tokens, layer):
    """Measure how much each head is utilized in practice.

    Heads with near-uniform attention or tiny output norms are underutilized.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with per-head utilization metrics.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]

    results = []
    for h in range(n_heads):
        P = patterns[h]  # [seq, seq]

        # Attention selectivity: how far from uniform?
        uniform = jnp.ones_like(P) / seq_len
        kl = float(jnp.mean(jnp.sum(P * jnp.log(jnp.maximum(P, 1e-10) / uniform), axis=-1)))

        # Output norm
        z_h = z[:, h, :]  # [seq, d_head]
        W_O_h = model.blocks[layer].attn.W_O[h]  # [d_head, d_model]
        output = z_h @ W_O_h  # [seq, d_model]
        output_norm = float(jnp.mean(jnp.linalg.norm(output, axis=-1)))

        # Utilization: combines selectivity and output magnitude
        selectivity = min(kl / max(jnp.log(jnp.array(seq_len, dtype=jnp.float32)), 1e-10), 1.0)
        utilization = float(selectivity)

        results.append({
            'head': h,
            'selectivity': float(selectivity),
            'output_norm': output_norm,
            'kl_from_uniform': float(kl),
            'utilization': utilization,
        })

    return {
        'layer': layer,
        'per_head': results,
        'mean_utilization': float(jnp.mean(jnp.array([r['utilization'] for r in results]))),
    }


def capacity_allocation(model, tokens):
    """Analyze how capacity is allocated across layers and heads.

    Returns:
        dict with capacity allocation summary.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Capacity proxy: output norm of each head
    head_norms = jnp.zeros((n_layers, n_heads))
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']  # [seq, n_heads, d_head]
        for h in range(n_heads):
            z_h = z[:, h, :]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h  # [seq, d_model]
            head_norms = head_norms.at[l, h].set(jnp.mean(jnp.linalg.norm(output, axis=-1)))

    total_capacity = float(jnp.sum(head_norms))

    # Per-layer allocation
    layer_share = jnp.sum(head_norms, axis=1) / max(total_capacity, 1e-10)

    # Find most/least important heads
    flat_norms = head_norms.reshape(-1)
    top_idx = int(jnp.argmax(flat_norms))
    bot_idx = int(jnp.argmin(flat_norms))

    return {
        'head_norms': [[float(head_norms[l, h]) for h in range(n_heads)] for l in range(n_layers)],
        'layer_share': [float(s) for s in layer_share],
        'total_capacity': total_capacity,
        'most_active_head': {'layer': top_idx // n_heads, 'head': top_idx % n_heads,
                             'norm': float(flat_norms[top_idx])},
        'least_active_head': {'layer': bot_idx // n_heads, 'head': bot_idx % n_heads,
                              'norm': float(flat_norms[bot_idx])},
        'gini_coefficient': float(_gini(flat_norms)),
    }


def _gini(values):
    """Compute Gini coefficient."""
    sorted_vals = jnp.sort(values)
    n = len(sorted_vals)
    index = jnp.arange(1, n + 1, dtype=jnp.float32)
    return float((2 * jnp.sum(index * sorted_vals) / (n * jnp.sum(sorted_vals) + 1e-10)) - (n + 1) / n)
