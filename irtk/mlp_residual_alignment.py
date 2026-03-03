"""MLP residual alignment: how MLP output relates to the residual stream."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def mlp_residual_cosine(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0) -> dict:
    """Cosine similarity between MLP output and residual stream at each position."""
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_mid", layer)]  # [seq, d_model] (after attn, before mlp)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]

    per_position = []
    for pos in range(len(tokens)):
        r = resid[pos]
        m = mlp_out[pos]
        r_norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        m_norm = jnp.sqrt(jnp.sum(m ** 2)).clip(1e-8)
        cos = float(jnp.sum(r * m) / (r_norm * m_norm))
        per_position.append({
            "position": pos,
            "cosine": cos,
            "is_reinforcing": cos > 0,
        })

    cosines = [p["cosine"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_cosine": sum(cosines) / len(cosines),
        "n_reinforcing": sum(1 for p in per_position if p["is_reinforcing"]),
    }


def mlp_parallel_perpendicular(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0, position: int = -1) -> dict:
    """Decompose MLP output into parallel and perpendicular to residual."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    resid = cache[("resid_mid", layer)][position]
    mlp_out = cache[("mlp_out", layer)][position]

    resid_norm = jnp.sqrt(jnp.sum(resid ** 2)).clip(1e-8)
    resid_dir = resid / resid_norm
    parallel_mag = float(jnp.sum(mlp_out * resid_dir))
    perp = mlp_out - parallel_mag * resid_dir
    perp_mag = float(jnp.sqrt(jnp.sum(perp ** 2)))
    mlp_norm = float(jnp.sqrt(jnp.sum(mlp_out ** 2)))

    return {
        "layer": layer,
        "position": position,
        "parallel_magnitude": parallel_mag,
        "perpendicular_magnitude": perp_mag,
        "total_magnitude": mlp_norm,
        "parallel_fraction": abs(parallel_mag) / max(mlp_norm, 1e-8),
        "is_reinforcing": parallel_mag > 0,
    }


def mlp_contribution_ratio(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0) -> dict:
    """Ratio of MLP output norm to residual stream norm."""
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_mid", layer)]
    mlp_out = cache[("mlp_out", layer)]

    per_position = []
    for pos in range(len(tokens)):
        r_norm = float(jnp.sqrt(jnp.sum(resid[pos] ** 2)))
        m_norm = float(jnp.sqrt(jnp.sum(mlp_out[pos] ** 2)))
        per_position.append({
            "position": pos,
            "residual_norm": r_norm,
            "mlp_norm": m_norm,
            "ratio": m_norm / max(r_norm, 1e-8),
        })

    ratios = [p["ratio"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_ratio": sum(ratios) / len(ratios),
    }


def mlp_unembed_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0, position: int = -1, top_k: int = 5) -> dict:
    """Which tokens does the MLP output promote (projected through W_U)?"""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    mlp_out = cache[("mlp_out", layer)][position]
    W_U = model.unembed.W_U
    logits = mlp_out @ W_U  # [d_vocab]

    top_indices = jnp.argsort(-logits)[:top_k]
    promoted = [(int(idx), float(logits[idx])) for idx in top_indices]
    bot_indices = jnp.argsort(logits)[:top_k]
    suppressed = [(int(idx), float(logits[idx])) for idx in bot_indices]

    return {
        "layer": layer,
        "position": position,
        "promoted": promoted,
        "suppressed": suppressed,
    }


def mlp_residual_alignment_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer MLP-residual alignment summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        cos = mlp_residual_cosine(model, tokens, layer)
        ratio = mlp_contribution_ratio(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_cosine": cos["mean_cosine"],
            "n_reinforcing": cos["n_reinforcing"],
            "mean_ratio": ratio["mean_ratio"],
        })
    return {"per_layer": per_layer}
