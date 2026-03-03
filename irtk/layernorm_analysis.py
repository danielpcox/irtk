"""Deep analysis of LayerNorm behavior in transformers.

Gain/bias decomposition, feature amplification/suppression patterns,
norm statistics across layers, directional effects, and LayerNorm
Jacobian analysis.
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


def gain_bias_decomposition(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
    component: str = "attn",
) -> dict:
    """Decompose LayerNorm effect into gain and bias contributions.

    LayerNorm(x) = gamma * (x - mean) / std + beta.
    This decomposes the output into the scaling (gamma) and shift (beta)
    contributions separately.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer index.
        pos: Position to analyze.
        component: "attn" for pre-attn LN, "mlp" for pre-MLP LN.

    Returns:
        Dict with gain_contribution, bias_contribution, normalized_input,
        scale_factor, effective_gain, effective_bias.
    """
    cache = _get_all_caches(model, tokens)

    # Get the pre-LN residual
    if component == "attn":
        resid_key = f"blocks.{layer}.hook_resid_pre"
    else:
        resid_key = f"blocks.{layer}.hook_resid_mid"

    if resid_key not in cache:
        return {"error": f"Key {resid_key} not found in cache"}

    x = np.array(cache[resid_key][pos])  # [d_model]

    # Get LN parameters
    if component == "attn":
        ln = model.blocks[layer].ln1
    else:
        ln = model.blocks[layer].ln2

    w = np.array(ln.w)  # [d_model]
    b = np.array(ln.b) if hasattr(ln, 'b') and ln.b is not None else np.zeros_like(w)

    # Compute LayerNorm manually
    mean_x = np.mean(x)
    var_x = np.var(x)
    eps = model.cfg.eps if hasattr(model.cfg, 'eps') else 1e-5
    std_x = np.sqrt(var_x + eps)

    normalized = (x - mean_x) / std_x
    gain_contribution = w * normalized
    bias_contribution = b

    output = gain_contribution + bias_contribution

    return {
        "gain_contribution": jnp.array(gain_contribution),
        "bias_contribution": jnp.array(bias_contribution),
        "normalized_input": jnp.array(normalized),
        "scale_factor": float(std_x),
        "input_mean": float(mean_x),
        "input_std": float(std_x),
        "gain_norm": float(np.linalg.norm(gain_contribution)),
        "bias_norm": float(np.linalg.norm(bias_contribution)),
        "output": jnp.array(output),
    }


def feature_amplification(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Identify which features are amplified vs suppressed by LayerNorm.

    Compares pre-LN and post-LN activation magnitudes per dimension
    to find features that gain or lose relative importance.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer index.
        pos: Position.
        top_k: Top features to report.

    Returns:
        Dict with amplified_dims, suppressed_dims, amplification_ratios,
        mean_amplification.
    """
    cache = _get_all_caches(model, tokens)

    resid_key = f"blocks.{layer}.hook_resid_pre"
    if resid_key not in cache:
        return {"error": "Residual not found"}

    x = np.array(cache[resid_key][pos])

    # Get LN params
    ln = model.blocks[layer].ln1
    w = np.array(ln.w)
    b = np.array(ln.b) if hasattr(ln, 'b') and ln.b is not None else np.zeros_like(w)

    mean_x = np.mean(x)
    var_x = np.var(x)
    eps = model.cfg.eps if hasattr(model.cfg, 'eps') else 1e-5
    std_x = np.sqrt(var_x + eps)
    normalized = (x - mean_x) / std_x
    output = w * normalized + b

    # Amplification ratio: |output[i]| / |input[i]|
    input_abs = np.abs(x)
    output_abs = np.abs(output)
    ratios = output_abs / (input_abs + 1e-10)

    # Amplified: ratio > 1
    amp_indices = np.argsort(ratios)[::-1][:top_k]
    amplified = [(int(i), float(ratios[i])) for i in amp_indices]

    # Suppressed: ratio < 1
    sup_indices = np.argsort(ratios)[:top_k]
    suppressed = [(int(i), float(ratios[i])) for i in sup_indices]

    return {
        "amplified_dims": amplified,
        "suppressed_dims": suppressed,
        "amplification_ratios": jnp.array(ratios),
        "mean_amplification": float(np.mean(ratios)),
        "n_amplified": int(np.sum(ratios > 1)),
        "n_suppressed": int(np.sum(ratios < 1)),
    }


def norm_statistics(
    model,
    tokens,
    layers: Optional[list] = None,
) -> dict:
    """Collect norm statistics across layers.

    Tracks input mean, variance, scale factors, and output norms
    through LayerNorms at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per_layer statistics, norm_growth_trend,
        variance_evolution.
    """
    cache = _get_all_caches(model, tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        resid_key = f"blocks.{l}.hook_resid_pre"
        if resid_key not in cache:
            continue

        resid = np.array(cache[resid_key])  # [seq, d_model]

        # Per-position statistics
        norms = np.linalg.norm(resid, axis=-1)  # [seq]
        means = np.mean(resid, axis=-1)          # [seq]
        variances = np.var(resid, axis=-1)       # [seq]

        per_layer.append({
            "layer": l,
            "mean_norm": float(np.mean(norms)),
            "max_norm": float(np.max(norms)),
            "mean_variance": float(np.mean(variances)),
            "mean_mean": float(np.mean(np.abs(means))),
        })

    # Trend
    if len(per_layer) > 1:
        norms_seq = [p["mean_norm"] for p in per_layer]
        trend = float(np.polyfit(range(len(norms_seq)), norms_seq, 1)[0])
    else:
        trend = 0.0

    variances_seq = [p["mean_variance"] for p in per_layer]

    return {
        "per_layer": per_layer,
        "norm_growth_trend": trend,
        "variance_evolution": variances_seq,
    }


def directional_effects(
    model,
    tokens,
    direction: Optional[np.ndarray] = None,
    layer: int = 0,
    pos: int = -1,
) -> dict:
    """Analyze how LayerNorm affects a specific direction in residual space.

    Given a direction vector, measures how much the projection onto
    that direction changes after LayerNorm.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        direction: Direction to analyze (default: unembed of most likely token).
        layer: Layer index.
        pos: Position.

    Returns:
        Dict with pre_projection, post_projection, projection_change,
        direction_preservation (cosine between pre and post projected).
    """
    cache = _get_all_caches(model, tokens)

    resid_key = f"blocks.{layer}.hook_resid_pre"
    if resid_key not in cache:
        return {"error": "Residual not found"}

    x = np.array(cache[resid_key][pos])

    # Default direction: first row of W_U
    if direction is None:
        direction = np.array(model.unembed.W_U[:, 0])
    else:
        direction = np.array(direction)

    d_normed = direction / (np.linalg.norm(direction) + 1e-10)

    # Pre-LN projection
    pre_proj = float(np.dot(x, d_normed))

    # Post-LN
    ln = model.blocks[layer].ln1
    w = np.array(ln.w)
    b = np.array(ln.b) if hasattr(ln, 'b') and ln.b is not None else np.zeros_like(w)
    mean_x = np.mean(x)
    eps = model.cfg.eps if hasattr(model.cfg, 'eps') else 1e-5
    std_x = np.sqrt(np.var(x) + eps)
    output = w * (x - mean_x) / std_x + b

    post_proj = float(np.dot(output, d_normed))

    # Direction preservation
    x_normed = x / (np.linalg.norm(x) + 1e-10)
    out_normed = output / (np.linalg.norm(output) + 1e-10)
    preservation = float(np.dot(x_normed, out_normed))

    return {
        "pre_projection": pre_proj,
        "post_projection": post_proj,
        "projection_change": post_proj - pre_proj,
        "projection_ratio": post_proj / (pre_proj + 1e-10),
        "direction_preservation": preservation,
    }


def layernorm_jacobian(
    model,
    tokens,
    layer: int = 0,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Compute the effective Jacobian of LayerNorm.

    The Jacobian reveals how small changes in each input dimension
    affect each output dimension through the normalization.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer index.
        pos: Position.
        top_k: Top eigenvalues/vectors to return.

    Returns:
        Dict with jacobian_norm, top_eigenvalues, effective_rank,
        condition_number.
    """
    cache = _get_all_caches(model, tokens)

    resid_key = f"blocks.{layer}.hook_resid_pre"
    if resid_key not in cache:
        return {"error": "Residual not found"}

    x = np.array(cache[resid_key][pos])
    d = len(x)

    ln = model.blocks[layer].ln1
    w = np.array(ln.w)
    eps = model.cfg.eps if hasattr(model.cfg, 'eps') else 1e-5

    mean_x = np.mean(x)
    var_x = np.var(x)
    std_x = np.sqrt(var_x + eps)

    # Analytical Jacobian of LayerNorm
    # d(LN)/dx_j = w_i * (delta_ij - 1/d) / std - w_i * (x_j - mean) * (x_i - mean) / (d * std^3)
    x_centered = x - mean_x
    I = np.eye(d)
    J = np.diag(w) @ (I / std_x - np.ones((d, d)) / (d * std_x)
                       - np.outer(x_centered, x_centered) / (d * std_x ** 3))

    # Compute properties
    jac_norm = float(np.linalg.norm(J, 'fro'))

    # SVD for eigenvalue analysis
    s = np.linalg.svd(J, compute_uv=False)
    top_k_actual = min(top_k, len(s))

    effective_rank = float(np.exp(-np.sum(
        (s / (np.sum(s) + 1e-10)) * np.log(s / (np.sum(s) + 1e-10) + 1e-10)
    )))
    condition = float(s[0] / (s[-1] + 1e-10))

    return {
        "jacobian_norm": jac_norm,
        "top_singular_values": s[:top_k_actual].tolist(),
        "effective_rank": effective_rank,
        "condition_number": condition,
        "d_model": d,
    }
