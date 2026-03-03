"""Attention distance analysis: how far attention reaches.

Tools for analyzing the effective distance of attention patterns:
- Mean attention distance per head
- Local vs global attention characterization
- Distance-weighted information flow
- Positional attention decay curves
- Receptive field estimation
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def mean_attention_distance(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
) -> dict:
    """Compute mean attention distance for each head.

    For each head, computes the expected distance between query and key
    positions weighted by attention probability.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per-head mean distances.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    seq_len = len(tokens)
    per_head = []

    for l in layers:
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]
        for h in range(model.cfg.n_heads):
            # Distance matrix: |query_pos - key_pos|
            positions = jnp.arange(seq_len)
            dist_matrix = jnp.abs(positions[:, None] - positions[None, :])  # [seq, seq]

            # Weighted average distance
            weighted_dist = pattern[h] * dist_matrix  # [seq, seq]
            per_query_dist = jnp.sum(weighted_dist, axis=-1)  # [seq]
            mean_dist = float(jnp.mean(per_query_dist[1:]))  # Skip pos 0 (only self)

            per_head.append({
                'layer': l,
                'head': h,
                'mean_distance': round(mean_dist, 4),
                'max_distance': round(float(jnp.max(per_query_dist)), 4),
            })

    distances = [h['mean_distance'] for h in per_head]
    return {
        'per_head': per_head,
        'overall_mean_distance': round(float(np.mean(distances)), 4) if distances else 0.0,
    }


def local_vs_global_heads(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    local_window: int = 3,
    layers: Optional[list[int]] = None,
) -> dict:
    """Classify heads as local (attend nearby) or global (attend far).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        local_window: Positions within this distance count as "local".
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per-head local/global classification.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    seq_len = len(tokens)
    per_head = []

    for l in layers:
        pattern = cache[f'blocks.{l}.attn.hook_pattern']

        for h in range(model.cfg.n_heads):
            positions = jnp.arange(seq_len)
            dist_matrix = jnp.abs(positions[:, None] - positions[None, :])
            local_mask = (dist_matrix <= local_window).astype(jnp.float32)

            local_mass = float(jnp.mean(jnp.sum(pattern[h] * local_mask, axis=-1)))
            global_mass = 1.0 - local_mass

            is_local = local_mass > 0.7
            is_global = global_mass > 0.5

            per_head.append({
                'layer': l,
                'head': h,
                'local_attention_mass': round(local_mass, 4),
                'global_attention_mass': round(global_mass, 4),
                'classification': 'local' if is_local else ('global' if is_global else 'mixed'),
            })

    classifications = [h['classification'] for h in per_head]
    return {
        'per_head': per_head,
        'n_local': classifications.count('local'),
        'n_global': classifications.count('global'),
        'n_mixed': classifications.count('mixed'),
    }


def distance_weighted_flow(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    source_pos: int = 0,
    layers: Optional[list[int]] = None,
) -> dict:
    """Compute how information from a source position spreads by distance.

    Tracks how much attention each position receives from the source,
    broken down by distance.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        source_pos: Source position to track flow from.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per-layer flow profiles.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    seq_len = len(tokens)
    per_layer = []

    for l in layers:
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]

        # How much does each position attend to the source?
        # Average across heads
        attn_to_source = np.array(jnp.mean(pattern[:, :, source_pos], axis=0))  # [seq]

        # Break down by distance from source
        distances = []
        for pos in range(seq_len):
            if pos > source_pos:  # Causal: only later positions attend to source
                dist = pos - source_pos
                distances.append({
                    'distance': dist,
                    'attention_to_source': round(float(attn_to_source[pos]), 4),
                })

        per_layer.append({
            'layer': l,
            'flow_by_distance': distances,
            'total_attention_received': round(float(np.sum(attn_to_source[source_pos + 1:])), 4),
        })

    return {
        'source_position': source_pos,
        'per_layer': per_layer,
    }


def attention_decay_curve(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    query_pos: int = -1,
    layers: Optional[list[int]] = None,
) -> dict:
    """Measure how attention decays with distance from the query.

    For a specific query position, plots attention weight as a function
    of distance to the key position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        query_pos: Query position to analyze.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with decay curves per head.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    seq_len = len(tokens)
    actual_pos = query_pos if query_pos >= 0 else seq_len + query_pos

    per_head = []
    for l in layers:
        pattern = cache[f'blocks.{l}.attn.hook_pattern']

        for h in range(model.cfg.n_heads):
            attn_row = np.array(pattern[h, actual_pos, :actual_pos + 1])

            # Decay by distance
            decay_points = []
            for k_pos in range(actual_pos + 1):
                dist = actual_pos - k_pos
                decay_points.append({
                    'distance': dist,
                    'attention': round(float(attn_row[k_pos]), 4),
                })

            # Compute half-life: distance at which attention drops to half of max
            max_attn = float(np.max(attn_row))
            half_life = -1
            if max_attn > 0.01:
                for dp in sorted(decay_points, key=lambda x: x['distance']):
                    if dp['attention'] >= max_attn / 2:
                        half_life = dp['distance']

            per_head.append({
                'layer': l,
                'head': h,
                'decay_curve': decay_points,
                'half_life': half_life,
                'max_attention': round(max_attn, 4),
            })

    return {
        'query_position': actual_pos,
        'per_head': per_head,
    }


def receptive_field(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_pos: int = -1,
    threshold: float = 0.05,
) -> dict:
    """Estimate the effective receptive field at a target position.

    Traces back through all layers to find which input positions
    have significant indirect influence on the target position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        target_pos: Position to analyze.
        threshold: Minimum attention weight to count as "in field".

    Returns:
        Dict with receptive field positions and widths per layer.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    actual_pos = target_pos if target_pos >= 0 else seq_len + target_pos

    per_layer = []
    for l in range(model.cfg.n_layers):
        pattern = cache[f'blocks.{l}.attn.hook_pattern']
        # Average across heads
        avg_pattern = np.array(jnp.mean(pattern, axis=0))  # [seq, seq]
        attn_row = avg_pattern[actual_pos, :]  # [seq]

        # Which positions are in the receptive field?
        in_field = np.where(attn_row >= threshold)[0]
        field_width = len(in_field)
        field_start = int(in_field[0]) if len(in_field) > 0 else actual_pos
        field_end = int(in_field[-1]) if len(in_field) > 0 else actual_pos

        per_layer.append({
            'layer': l,
            'field_width': field_width,
            'field_start': field_start,
            'field_end': field_end,
            'positions_in_field': [int(p) for p in in_field],
        })

    return {
        'target_position': actual_pos,
        'threshold': threshold,
        'per_layer': per_layer,
        'max_field_width': max(p['field_width'] for p in per_layer) if per_layer else 0,
    }
