"""Token mixing analysis for mechanistic interpretability.

How tokens mix across positions through attention: mixing matrix
computation, mixing speed across layers, position-wise information
spread, self-information retention, and mixing entropy.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def _get_all_caches(model, tokens):
    """Run model and return full cache."""
    from irtk.hook_points import HookState
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)
    return cache


def mixing_matrix(
    model,
    tokens,
    layer: int = 0,
) -> dict:
    """Compute the effective token mixing matrix at a given layer.

    The mixing matrix M[i,j] measures how much information from
    position j contributes to position i's representation after attention.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer to analyze.

    Returns:
        Dict with mixing_matrix, diagonal (self-mixing), off_diagonal_mean,
        strongest_mixing_pair.
    """
    cache = _get_all_caches(model, tokens)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"error": "Attention pattern not found"}

    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    # Average across heads for the effective mixing
    mix = np.mean(patterns, axis=0)  # [seq, seq]

    diagonal = np.diag(mix)
    off_diag = mix.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_mean = float(np.sum(off_diag) / max(mix.shape[0] * (mix.shape[0] - 1), 1))

    # Strongest mixing pair (excluding self)
    off_diag_flat = off_diag.flatten()
    max_idx = np.argmax(off_diag_flat)
    max_i, max_j = divmod(int(max_idx), mix.shape[1])

    return {
        "mixing_matrix": jnp.array(mix),
        "diagonal": jnp.array(diagonal),
        "off_diagonal_mean": off_diag_mean,
        "strongest_mixing_pair": (max_i, max_j, float(mix[max_i, max_j])),
        "self_mixing_mean": float(np.mean(diagonal)),
    }


def mixing_speed(
    model,
    tokens,
) -> dict:
    """Measure how quickly tokens mix across layers.

    Tracks the off-diagonal mass of the mixing matrix (how much
    information flows between positions) at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.

    Returns:
        Dict with per_layer mixing fraction, mixing_acceleration,
        full_mixing_layer (where off-diag > diag).
    """
    cache = _get_all_caches(model, tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])
        mix = np.mean(patterns, axis=0)

        diagonal_mean = float(np.mean(np.diag(mix)))
        off_diag = mix.copy()
        np.fill_diagonal(off_diag, 0)
        off_diag_fraction = float(np.sum(off_diag) / (np.sum(mix) + 1e-10))

        per_layer.append({
            "layer": l,
            "self_retention": diagonal_mean,
            "mixing_fraction": off_diag_fraction,
        })

    # Find where mixing exceeds self-retention
    full_mixing_layer = -1
    for p in per_layer:
        if p["mixing_fraction"] > 0.5:
            full_mixing_layer = p["layer"]
            break

    fracs = [p["mixing_fraction"] for p in per_layer]
    if len(fracs) > 1:
        acceleration = float(np.diff(fracs).mean())
    else:
        acceleration = 0.0

    return {
        "per_layer": per_layer,
        "mixing_acceleration": acceleration,
        "full_mixing_layer": full_mixing_layer,
    }


def information_spread(
    model,
    tokens,
    source_pos: int = 0,
    layer: int = 0,
) -> dict:
    """Measure how information from a source position spreads.

    Tracks how much each other position receives from the source.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        source_pos: Source position.
        layer: Layer to analyze.

    Returns:
        Dict with spread_vector (attention from all positions to source),
        reach (how many positions receive > threshold), entropy.
    """
    cache = _get_all_caches(model, tokens)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"error": "Attention pattern not found"}

    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    seq_len = patterns.shape[1]

    # How much each position attends to source_pos (averaged across heads)
    spread = np.mean(patterns[:, :, source_pos], axis=0)  # [seq]

    # Reach: positions receiving significant attention from this source
    threshold = 1.0 / seq_len  # above uniform
    reach = int(np.sum(spread > threshold))

    # Entropy of the spread
    spread_normalized = spread / (np.sum(spread) + 1e-10)
    entropy = -float(np.sum(spread_normalized * np.log(spread_normalized + 1e-10)))

    return {
        "spread_vector": jnp.array(spread),
        "reach": reach,
        "entropy": entropy,
        "max_receiver": int(np.argmax(spread)),
        "mean_spread": float(np.mean(spread)),
    }


def self_information_retention(
    model,
    tokens,
    layers: Optional[list] = None,
) -> dict:
    """Track how much each position retains its own information.

    Measures the self-attention (diagonal of mixing matrix) across layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layers: Layers to analyze.

    Returns:
        Dict with retention_per_layer, per_position_retention,
        most_retained, least_retained.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    retention_matrix = np.zeros((len(layers), seq_len))

    for li, l in enumerate(layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue
        patterns = np.array(cache[pattern_key])
        # Self-attention per position, averaged across heads
        for pos in range(seq_len):
            retention_matrix[li, pos] = np.mean(patterns[:, pos, pos])

    # Per-layer average retention
    layer_retention = np.mean(retention_matrix, axis=1)

    # Per-position average retention (across layers)
    pos_retention = np.mean(retention_matrix, axis=0)

    most_retained = int(np.argmax(pos_retention))
    least_retained = int(np.argmin(pos_retention))

    return {
        "retention_per_layer": jnp.array(layer_retention),
        "per_position_retention": jnp.array(pos_retention),
        "retention_matrix": jnp.array(retention_matrix),
        "most_retained": most_retained,
        "least_retained": least_retained,
    }


def mixing_entropy(
    model,
    tokens,
    layers: Optional[list] = None,
) -> dict:
    """Compute entropy of the mixing distribution at each layer.

    High entropy = uniform mixing (all positions contribute equally).
    Low entropy = focused mixing (few positions dominate).

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layers: Layers to analyze.

    Returns:
        Dict with per_layer_entropy, entropy_trend,
        most_uniform_layer, most_focused_layer.
    """
    cache = _get_all_caches(model, tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])
        mix = np.mean(patterns, axis=0)  # [seq, seq]

        # Entropy per query position
        entropies = -np.sum(mix * np.log(mix + 1e-10), axis=-1)
        mean_entropy = float(np.mean(entropies))
        max_entropy = float(np.log(mix.shape[1]))

        per_layer.append({
            "layer": l,
            "mean_entropy": mean_entropy,
            "max_possible_entropy": max_entropy,
            "normalized_entropy": mean_entropy / (max_entropy + 1e-10),
        })

    if per_layer:
        entropies = [p["mean_entropy"] for p in per_layer]
        most_uniform = max(per_layer, key=lambda x: x["mean_entropy"])["layer"]
        most_focused = min(per_layer, key=lambda x: x["mean_entropy"])["layer"]
        trend = float(np.polyfit(range(len(entropies)), entropies, 1)[0]) if len(entropies) > 1 else 0.0
    else:
        most_uniform = most_focused = 0
        trend = 0.0

    return {
        "per_layer": per_layer,
        "entropy_trend": trend,
        "most_uniform_layer": most_uniform,
        "most_focused_layer": most_focused,
    }
