"""Cross-position attention: analyze information flow between positions."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def position_information_flow(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How much information flows between each pair of positions?

    Uses attention weights summed across heads as a proxy for info flow.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    patterns = cache[pattern_key]  # [n_heads, seq, seq]
    seq_len = patterns.shape[1]

    # Average flow across heads
    avg_flow = jnp.mean(patterns, axis=0)  # [seq, seq]

    per_position = []
    for pos in range(seq_len):
        # How much does this position receive from others?
        received = float(jnp.sum(avg_flow[pos, :pos]))  # exclude self
        # How much does this position send to later positions?
        sent = float(jnp.sum(avg_flow[pos+1:, pos])) if pos < seq_len - 1 else 0.0

        per_position.append({
            'position': pos,
            'total_received': received,
            'total_sent': sent,
            'is_hub': received > 0.5 and sent > 0.5,
        })

    return {
        'layer': layer,
        'per_position': per_position,
        'n_hubs': sum(1 for p in per_position if p['is_hub']),
    }


def source_position_importance(model: HookedTransformer, tokens: jnp.ndarray, layer: int, target_position: int = -1) -> dict:
    """Which source positions are most important for the target?

    Ranks source positions by attention weight from the target.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    target = target_position if target_position >= 0 else seq_len + target_position
    n_heads = model.cfg.n_heads

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    patterns = cache[pattern_key]  # [n_heads, seq, seq]

    per_source = []
    for src in range(target + 1):
        # Sum attention across heads
        total_attn = float(jnp.sum(patterns[:, target, src]))
        per_head_attn = [float(patterns[h, target, src]) for h in range(n_heads)]
        max_head = int(jnp.argmax(jnp.array(per_head_attn)))

        per_source.append({
            'source_position': src,
            'source_token': int(tokens[src]),
            'total_attention': total_attn,
            'max_head': max_head,
            'max_head_attention': per_head_attn[max_head],
        })

    per_source.sort(key=lambda x: x['total_attention'], reverse=True)

    return {
        'layer': layer,
        'target_position': target,
        'per_source': per_source,
        'top_source': per_source[0]['source_position'] if per_source else 0,
    }


def attention_flow_matrix(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Compute a full-model attention flow matrix.

    Approximates total information flow by averaging attention across all
    layers and heads.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    # Accumulate flow
    total_flow = jnp.zeros((seq_len, seq_len))
    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        patterns = cache[pattern_key]
        layer_flow = jnp.mean(patterns, axis=0)  # [seq, seq]
        total_flow = total_flow + layer_flow

    total_flow = total_flow / n_layers

    # Summary stats
    per_position = []
    for pos in range(seq_len):
        incoming = float(jnp.sum(total_flow[pos]))
        max_source = int(jnp.argmax(total_flow[pos]))
        per_position.append({
            'position': pos,
            'total_incoming_flow': incoming,
            'primary_source': max_source,
        })

    return {
        'per_position': per_position,
        'mean_flow': float(jnp.mean(total_flow)),
    }


def position_pair_interaction(model: HookedTransformer, tokens: jnp.ndarray, pos_a: int, pos_b: int) -> dict:
    """Detailed analysis of attention interaction between two positions.

    Shows per-layer, per-head attention from pos_b to pos_a.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Ensure pos_b > pos_a (causal attention)
    if pos_b < pos_a:
        pos_a, pos_b = pos_b, pos_a

    per_layer = []
    total_attn = 0.0

    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        patterns = cache[pattern_key]

        per_head = []
        layer_total = 0.0
        for h in range(n_heads):
            attn = float(patterns[h, pos_b, pos_a])
            per_head.append({'head': h, 'attention': attn})
            layer_total += attn

        per_layer.append({
            'layer': layer,
            'per_head': per_head,
            'layer_total': layer_total,
            'max_head': max(per_head, key=lambda x: x['attention'])['head'],
        })
        total_attn += layer_total

    return {
        'pos_a': pos_a,
        'pos_b': pos_b,
        'per_layer': per_layer,
        'total_attention': total_attn,
        'mean_per_layer': total_attn / n_layers,
    }


def attention_bottleneck_positions(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Find positions that are bottlenecks for information flow.

    A bottleneck is a position that many later positions attend to heavily.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    # Accumulate how much each position is attended to
    incoming = jnp.zeros(seq_len)

    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        patterns = cache[pattern_key]
        avg_pattern = jnp.mean(patterns, axis=0)  # [seq, seq]
        incoming = incoming + jnp.sum(avg_pattern, axis=0)  # sum across query positions

    incoming = incoming / n_layers

    per_position = []
    for pos in range(seq_len):
        per_position.append({
            'position': pos,
            'token': int(tokens[pos]),
            'incoming_attention': float(incoming[pos]),
            'is_bottleneck': bool(float(incoming[pos]) > float(jnp.mean(incoming) + jnp.std(incoming))),
        })

    per_position.sort(key=lambda x: x['incoming_attention'], reverse=True)

    return {
        'per_position': per_position,
        'n_bottlenecks': sum(1 for p in per_position if p['is_bottleneck']),
        'top_bottleneck': per_position[0]['position'] if per_position else 0,
    }
