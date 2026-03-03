"""Attention residual decomposition: how attention modifies the residual stream."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def attention_parallel_perpendicular(model: HookedTransformer, tokens: jnp.ndarray,
                                        layer: int = 0, position: int = -1) -> dict:
    """Decompose attention output into parallel and perpendicular to residual.

    Parallel = reinforcing current direction; perpendicular = adding new info.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    resid = cache[("resid_pre", layer)][position]
    attn_out = cache[("attn_out", layer)][position]

    resid_norm = jnp.sqrt(jnp.sum(resid ** 2)).clip(1e-8)
    resid_dir = resid / resid_norm
    parallel_mag = float(jnp.sum(attn_out * resid_dir))
    parallel = parallel_mag * resid_dir
    perp = attn_out - parallel
    perp_mag = float(jnp.sqrt(jnp.sum(perp ** 2)))
    attn_norm = float(jnp.sqrt(jnp.sum(attn_out ** 2)))

    return {
        "layer": layer,
        "position": position,
        "parallel_magnitude": parallel_mag,
        "perpendicular_magnitude": perp_mag,
        "total_magnitude": attn_norm,
        "parallel_fraction": abs(parallel_mag) / max(attn_norm, 1e-8),
        "is_reinforcing": parallel_mag > 0,
    }


def per_head_residual_decomposition(model: HookedTransformer, tokens: jnp.ndarray,
                                       layer: int = 0, position: int = -1) -> dict:
    """Per-head parallel/perpendicular decomposition relative to residual."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    resid = cache[("resid_pre", layer)][position]
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    resid_norm = jnp.sqrt(jnp.sum(resid ** 2)).clip(1e-8)
    resid_dir = resid / resid_norm

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[position, head, :] @ W_O[head]
        par = float(jnp.sum(head_out * resid_dir))
        head_norm = float(jnp.sqrt(jnp.sum(head_out ** 2)))
        per_head.append({
            "head": int(head),
            "parallel": par,
            "perpendicular": float(jnp.sqrt(max(head_norm ** 2 - par ** 2, 0))),
            "total_norm": head_norm,
            "is_reinforcing": par > 0,
        })
    return {
        "layer": layer,
        "position": position,
        "per_head": per_head,
        "n_reinforcing": sum(1 for h in per_head if h["is_reinforcing"]),
    }


def attention_update_angle(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0) -> dict:
    """Angle between attention output and residual stream at each position."""
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_pre", layer)]  # [seq, d_model]
    attn = cache[("attn_out", layer)]  # [seq, d_model]

    per_position = []
    for pos in range(len(tokens)):
        r = resid[pos]
        a = attn[pos]
        r_norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        a_norm = jnp.sqrt(jnp.sum(a ** 2)).clip(1e-8)
        cos = float(jnp.sum(r * a) / (r_norm * a_norm))
        cos = max(-1.0, min(1.0, cos))
        per_position.append({
            "position": pos,
            "cosine": cos,
            "angle_degrees": float(jnp.arccos(jnp.array(cos)) * 180 / jnp.pi),
        })

    angles = [p["angle_degrees"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_angle": sum(angles) / len(angles),
    }


def attention_residual_ratio(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0) -> dict:
    """Ratio of attention output norm to residual stream norm."""
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_pre", layer)]
    attn = cache[("attn_out", layer)]

    per_position = []
    for pos in range(len(tokens)):
        r_norm = float(jnp.sqrt(jnp.sum(resid[pos] ** 2)))
        a_norm = float(jnp.sqrt(jnp.sum(attn[pos] ** 2)))
        per_position.append({
            "position": pos,
            "residual_norm": r_norm,
            "attention_norm": a_norm,
            "ratio": a_norm / max(r_norm, 1e-8),
        })

    ratios = [p["ratio"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_ratio": sum(ratios) / len(ratios),
    }


def attention_residual_decomposition_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                                position: int = -1) -> dict:
    """Cross-layer attention-residual decomposition summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        pp = attention_parallel_perpendicular(model, tokens, layer, position)
        angle = attention_update_angle(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "parallel_fraction": pp["parallel_fraction"],
            "is_reinforcing": pp["is_reinforcing"],
            "mean_angle": angle["mean_angle"],
        })
    return {"per_layer": per_layer, "position": per_layer[0].get("layer", position)}
