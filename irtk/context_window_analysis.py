"""Context window utilization analysis.

Analyze how models use their context window: effective context length,
attention decay profiles, position-dependent capability, context boundary
effects, and information horizon estimation.
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


def effective_context_length(
    model,
    tokens,
    threshold: float = 0.01,
) -> dict:
    """Estimate the effective context length used by the model.

    Measures how far back in the context the model actually attends,
    versus the theoretical maximum context length.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        threshold: Minimum attention weight to consider as "used".

    Returns:
        Dict with effective_length per layer, overall_effective_length,
        utilization_ratio.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]

        # For the last position, measure how far back attention goes
        last_pos_attn = np.mean(patterns[:, -1, :], axis=0)  # [seq]

        # Effective length: furthest position with attention > threshold
        active_positions = np.where(last_pos_attn > threshold)[0]
        if len(active_positions) > 0:
            eff_len = int(seq_len - active_positions[0])
        else:
            eff_len = 0

        per_layer.append({
            "layer": l,
            "effective_length": eff_len,
            "utilization": eff_len / seq_len if seq_len > 0 else 0,
        })

    overall = max(p["effective_length"] for p in per_layer) if per_layer else 0

    return {
        "per_layer": per_layer,
        "overall_effective_length": overall,
        "utilization_ratio": overall / seq_len if seq_len > 0 else 0,
        "max_context_length": seq_len,
    }


def attention_decay_profile(
    model,
    tokens,
    query_pos: int = -1,
) -> dict:
    """Profile how attention decays with distance from the query.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        query_pos: Query position to analyze.

    Returns:
        Dict with per_layer decay curves, mean_decay_rate,
        half_life (distance at which attention halves).
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if query_pos < 0:
        query_pos = seq_len + query_pos

    per_layer = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])
        avg_attn = np.mean(patterns[:, query_pos, :query_pos + 1], axis=0)

        # Compute distance-based decay
        distances = query_pos - np.arange(query_pos + 1)
        if len(distances) > 1:
            # Average attention at each distance
            decay_curve = avg_attn[::-1]  # reorder by distance (0 = self, 1 = prev, ...)

            # Half-life: distance at which attention drops below half of self
            self_attn = decay_curve[0]
            half_indices = np.where(decay_curve < self_attn * 0.5)[0]
            half_life = int(half_indices[0]) if len(half_indices) > 0 else len(decay_curve)
        else:
            decay_curve = avg_attn
            half_life = 0

        per_layer.append({
            "layer": l,
            "decay_curve": decay_curve.tolist(),
            "half_life": half_life,
            "self_attention": float(avg_attn[query_pos]) if query_pos < len(avg_attn) else 0.0,
        })

    half_lives = [p["half_life"] for p in per_layer]
    mean_half_life = float(np.mean(half_lives)) if half_lives else 0

    return {
        "per_layer": per_layer,
        "mean_half_life": mean_half_life,
    }


def position_dependent_capability(
    model,
    tokens,
) -> dict:
    """Analyze how model capability varies with position in the context.

    Later positions have access to more context; measures how this
    affects prediction quality indicators.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.

    Returns:
        Dict with per_position logit entropy, norm, confidence,
        capability_trend.
    """
    logits = np.array(model(tokens))  # [seq, d_vocab]
    seq_len = logits.shape[0]

    per_position = []
    for pos in range(seq_len):
        # Softmax entropy
        log_probs = logits[pos] - np.max(logits[pos])
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

        # Confidence: max probability
        confidence = float(np.max(probs))

        # Logit norm
        norm = float(np.linalg.norm(logits[pos]))

        per_position.append({
            "position": pos,
            "entropy": entropy,
            "confidence": confidence,
            "logit_norm": norm,
        })

    confidences = [p["confidence"] for p in per_position]
    if len(confidences) > 1:
        trend = float(np.polyfit(range(len(confidences)), confidences, 1)[0])
    else:
        trend = 0.0

    return {
        "per_position": per_position,
        "capability_trend": trend,
        "mean_entropy": float(np.mean([p["entropy"] for p in per_position])),
        "mean_confidence": float(np.mean(confidences)),
    }


def context_boundary_effects(
    model,
    tokens,
    boundary_pos: Optional[int] = None,
) -> dict:
    """Analyze effects at context boundaries (e.g., sentence boundaries).

    Measures how attention patterns and activations change around
    a boundary position.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        boundary_pos: Position of the boundary (default: middle).

    Returns:
        Dict with attention_across_boundary, residual_discontinuity,
        pre_boundary_coherence, post_boundary_coherence.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)

    if boundary_pos is None:
        boundary_pos = seq_len // 2

    n_layers = model.cfg.n_layers

    # Attention flowing across boundary
    cross_boundary_attn = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])
        # Attention from post-boundary to pre-boundary
        post_to_pre = np.mean(patterns[:, boundary_pos:, :boundary_pos])
        # Attention within pre-boundary
        within_pre = np.mean(patterns[:, :boundary_pos, :boundary_pos])
        # Attention within post-boundary
        within_post = np.mean(patterns[:, boundary_pos:, boundary_pos:])

        cross_boundary_attn.append({
            "layer": l,
            "cross_boundary": float(post_to_pre),
            "within_pre": float(within_pre),
            "within_post": float(within_post),
        })

    # Residual discontinuity at boundary
    resid_discontinuity = []
    for l in range(n_layers):
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache and boundary_pos > 0:
            resid = np.array(cache[resid_key])
            pre = resid[boundary_pos - 1]
            post = resid[boundary_pos]
            disc = float(np.linalg.norm(post - pre))
            resid_discontinuity.append({"layer": l, "discontinuity": disc})

    return {
        "attention_across_boundary": cross_boundary_attn,
        "residual_discontinuity": resid_discontinuity,
        "boundary_pos": boundary_pos,
    }


def information_horizon(
    model,
    tokens,
    layer: int = -1,
) -> dict:
    """Estimate the information horizon at a given layer.

    The information horizon is the effective range over which
    a position can gather information from the context.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer to analyze (default: last).

    Returns:
        Dict with per_position horizon, mean_horizon,
        horizon_distribution.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)

    if layer < 0:
        layer = model.cfg.n_layers + layer

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"error": "Attention pattern not found"}

    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    avg_patterns = np.mean(patterns, axis=0)  # [seq, seq]

    per_position = []
    for pos in range(seq_len):
        attn = avg_patterns[pos, :pos + 1]
        if len(attn) <= 1:
            per_position.append({"position": pos, "horizon": 0})
            continue

        # Weighted average distance
        distances = pos - np.arange(pos + 1)
        weighted_dist = float(np.sum(attn * distances))

        # 90th percentile: distance covering 90% of attention
        cumsum = np.cumsum(attn[::-1])  # from nearest to farthest
        p90_idx = np.searchsorted(cumsum, 0.9)
        horizon_90 = int(min(p90_idx + 1, pos + 1))

        per_position.append({
            "position": pos,
            "weighted_distance": weighted_dist,
            "horizon_90": horizon_90,
        })

    horizons = [p.get("horizon_90", p.get("horizon", 0)) for p in per_position]

    return {
        "per_position": per_position,
        "mean_horizon": float(np.mean(horizons)),
        "max_horizon": int(np.max(horizons)) if horizons else 0,
    }
