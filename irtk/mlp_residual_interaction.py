"""MLP-residual stream interaction: how MLP outputs interact with the residual."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def mlp_residual_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                            layer: int = 0) -> dict:
    """Measure alignment between MLP output and residual stream.

    High alignment means MLP reinforces existing directions.
    Low/negative alignment means MLP pushes in new directions.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]  # [seq, d_model]
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    seq_len = resid_pre.shape[0]

    per_position = []
    for pos in range(seq_len):
        r = resid_pre[pos]
        m = mlp_out[pos]
        r_norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        m_norm = jnp.sqrt(jnp.sum(m ** 2)).clip(1e-8)
        cos = float(jnp.sum(r * m) / (r_norm * m_norm))
        per_position.append({
            "position": pos,
            "cosine": cos,
            "mlp_norm": float(m_norm),
            "residual_norm": float(r_norm),
        })
    cosines = [p["cosine"] for p in per_position]
    mean_cos = sum(cosines) / len(cosines)
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_alignment": mean_cos,
        "is_reinforcing": mean_cos > 0.3,
    }


def mlp_contribution_ratio(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Ratio of MLP output norm to residual norm across layers.

    Shows how much the MLP contributes relative to the existing
    residual stream at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid_pre = cache[("resid_pre", layer)]
        mlp_out = cache[("mlp_out", layer)]
        resid_norm = float(jnp.mean(jnp.sqrt(jnp.sum(resid_pre ** 2, axis=-1))))
        mlp_norm = float(jnp.mean(jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1))))
        ratio = mlp_norm / max(resid_norm, 1e-8)
        per_layer.append({
            "layer": layer,
            "residual_norm": resid_norm,
            "mlp_norm": mlp_norm,
            "contribution_ratio": ratio,
        })
    ratios = [p["contribution_ratio"] for p in per_layer]
    return {
        "per_layer": per_layer,
        "mean_ratio": sum(ratios) / len(ratios),
        "max_ratio_layer": int(jnp.argmax(jnp.array(ratios))),
    }


def mlp_residual_decomposition(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0, position: int = -1) -> dict:
    """Decompose MLP output into parallel and perpendicular components
    relative to the residual stream.

    The parallel component reinforces what's already there;
    the perpendicular component adds genuinely new information.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]
    mlp_out = cache[("mlp_out", layer)]
    r = resid_pre[position]
    m = mlp_out[position]

    r_norm_sq = jnp.sum(r ** 2).clip(1e-8)
    proj_scalar = jnp.sum(m * r) / r_norm_sq
    parallel = proj_scalar * r
    perpendicular = m - parallel

    parallel_norm = float(jnp.sqrt(jnp.sum(parallel ** 2)))
    perp_norm = float(jnp.sqrt(jnp.sum(perpendicular ** 2)))
    total_norm = float(jnp.sqrt(jnp.sum(m ** 2)))

    return {
        "layer": layer,
        "position": int(position % resid_pre.shape[0]),
        "parallel_norm": parallel_norm,
        "perpendicular_norm": perp_norm,
        "total_norm": total_norm,
        "parallel_fraction": parallel_norm / max(total_norm, 1e-8),
        "perpendicular_fraction": perp_norm / max(total_norm, 1e-8),
        "projection_scalar": float(proj_scalar),
    }


def mlp_vs_attention_balance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Compare MLP and attention output norms across layers.

    Shows the relative contribution balance: which component
    dominates at each layer?
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]
        attn_norm = float(jnp.mean(jnp.sqrt(jnp.sum(attn_out ** 2, axis=-1))))
        mlp_norm = float(jnp.mean(jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1))))
        total = attn_norm + mlp_norm
        per_layer.append({
            "layer": layer,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "attn_fraction": attn_norm / max(total, 1e-8),
            "mlp_fraction": mlp_norm / max(total, 1e-8),
            "dominant": "attention" if attn_norm > mlp_norm else "mlp",
        })
    return {
        "per_layer": per_layer,
        "attn_dominant_layers": sum(1 for p in per_layer if p["dominant"] == "attention"),
        "mlp_dominant_layers": sum(1 for p in per_layer if p["dominant"] == "mlp"),
    }


def mlp_residual_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Combined summary of MLP-residual interaction."""
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid_pre = cache[("resid_pre", layer)]
        mlp_out = cache[("mlp_out", layer)]
        attn_out = cache[("attn_out", layer)]

        # Alignment
        r_norms = jnp.sqrt(jnp.sum(resid_pre ** 2, axis=-1, keepdims=True)).clip(1e-8)
        m_norms = jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
        cos = jnp.sum((resid_pre / r_norms) * (mlp_out / m_norms), axis=-1)
        mean_cos = float(jnp.mean(cos))

        # Norms
        mlp_norm = float(jnp.mean(m_norms))
        attn_norm = float(jnp.mean(jnp.sqrt(jnp.sum(attn_out ** 2, axis=-1))))
        resid_norm = float(jnp.mean(r_norms))

        per_layer.append({
            "layer": layer,
            "mlp_alignment": mean_cos,
            "mlp_norm": mlp_norm,
            "attn_norm": attn_norm,
            "residual_norm": resid_norm,
            "mlp_ratio": mlp_norm / max(resid_norm, 1e-8),
        })
    return {"per_layer": per_layer}
