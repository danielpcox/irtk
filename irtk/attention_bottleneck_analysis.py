"""Attention bottleneck analysis: identify information bottlenecks in attention."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def attention_rank_bottleneck(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Effective rank of attention patterns per head — low rank = bottleneck."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        for head in range(n_heads):
            P = pattern[head]  # [seq, seq]
            sv = jnp.linalg.svd(P, compute_uv=False)
            sv_norm = sv / (jnp.sum(sv) + 1e-10)
            entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
            eff_rank = float(jnp.exp(entropy))

            per_head.append({
                'layer': layer,
                'head': head,
                'effective_rank': eff_rank,
                'top_sv_fraction': float(sv[0] / (jnp.sum(sv) + 1e-10)),
                'is_bottleneck': eff_rank < 2.0,
            })

    n_bottleneck = sum(1 for h in per_head if h['is_bottleneck'])
    return {
        'per_head': per_head,
        'n_bottleneck': n_bottleneck,
    }


def attention_information_throughput(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much information passes through each attention head?

    Measured by attention-weighted value vector norms.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]

        for head in range(n_heads):
            P = pattern[head]  # [seq, seq]
            V = v[:, head, :]  # [seq, d_head]
            v_norms = jnp.linalg.norm(V, axis=1)  # [seq]

            # Weighted value norm per query position
            weighted_norms = P @ v_norms  # [seq]
            mean_throughput = float(jnp.mean(weighted_norms))
            max_throughput = float(jnp.max(weighted_norms))

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_throughput': mean_throughput,
                'max_throughput': max_throughput,
                'mean_value_norm': float(jnp.mean(v_norms)),
            })

    return {
        'per_head': per_head,
    }


def attention_source_concentration(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How concentrated is each head's information source?

    High concentration = few source positions dominate.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        for head in range(n_heads):
            P = pattern[head]  # [seq, seq]
            # Average entropy across query positions
            entropies = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)
            mean_entropy = float(jnp.mean(entropies))
            max_entropy = float(jnp.log(seq_len))

            # Top-1 mass (averaged)
            top1_mass = float(jnp.mean(jnp.max(P, axis=1)))

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_entropy': mean_entropy,
                'normalized_entropy': mean_entropy / (max_entropy + 1e-10),
                'mean_top1_mass': top1_mass,
                'is_concentrated': top1_mass > 0.5,
            })

    n_concentrated = sum(1 for h in per_head if h['is_concentrated'])
    return {
        'per_head': per_head,
        'n_concentrated': n_concentrated,
    }


def attention_layer_bottleneck(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Per-layer summary of attention bottleneck severity."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_layer = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        layer_ranks = []
        layer_entropies = []

        for head in range(n_heads):
            P = pattern[head]
            sv = jnp.linalg.svd(P, compute_uv=False)
            sv_norm = sv / (jnp.sum(sv) + 1e-10)
            entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
            layer_ranks.append(float(jnp.exp(entropy)))

            attn_entropy = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)
            layer_entropies.append(float(jnp.mean(attn_entropy)))

        mean_rank = sum(layer_ranks) / len(layer_ranks)
        min_rank = min(layer_ranks)

        per_layer.append({
            'layer': layer,
            'mean_pattern_rank': mean_rank,
            'min_pattern_rank': min_rank,
            'mean_attn_entropy': sum(layer_entropies) / len(layer_entropies),
            'has_bottleneck_head': min_rank < 2.0,
        })

    return {
        'per_layer': per_layer,
    }


def attention_position_bottleneck(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Which query positions are bottlenecked (receive from few sources)?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    # Average over all heads
    per_position = []
    for pos in range(seq_len):
        entropies = []
        top1_masses = []
        for layer in range(n_layers):
            pattern = cache[f'blocks.{layer}.attn.hook_pattern']
            for head in range(n_heads):
                row = pattern[head, pos, :pos + 1]
                ent = float(-jnp.sum(row * jnp.log(row + 1e-10)))
                entropies.append(ent)
                top1_masses.append(float(jnp.max(row)))

        mean_ent = sum(entropies) / len(entropies)
        mean_top1 = sum(top1_masses) / len(top1_masses)

        per_position.append({
            'position': pos,
            'mean_entropy': mean_ent,
            'mean_top1_mass': mean_top1,
            'is_bottlenecked': mean_top1 > 0.7,
        })

    n_bottlenecked = sum(1 for p in per_position if p['is_bottlenecked'])
    return {
        'per_position': per_position,
        'n_bottlenecked': n_bottlenecked,
    }
