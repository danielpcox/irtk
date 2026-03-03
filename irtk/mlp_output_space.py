"""MLP output space analysis: properties of MLP layer outputs."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def mlp_output_rank(model: HookedTransformer, tokens: jnp.ndarray,
                     layer: int = 0) -> dict:
    """Effective rank of MLP output across positions.

    Low rank means the MLP compresses information into few directions;
    high rank means it uses many dimensions.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]

    svs = jnp.linalg.svd(mlp_out, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    # Variance explained
    cumvar = jnp.cumsum(svs ** 2) / jnp.sum(svs ** 2).clip(1e-8)
    dim_90 = int(jnp.searchsorted(cumvar, 0.9)) + 1

    return {
        "layer": layer,
        "effective_rank": eff_rank,
        "dim_for_90_pct": dim_90,
        "top_singular_value": float(svs[0]),
        "n_singular_values": int(len(svs)),
    }


def mlp_output_direction_analysis(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0, top_k: int = 5) -> dict:
    """Principal directions of MLP output using SVD."""
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]

    U, S, Vt = jnp.linalg.svd(mlp_out, full_matrices=False)
    total_var = jnp.sum(S ** 2).clip(1e-8)

    directions = []
    for i in range(min(top_k, len(S))):
        directions.append({
            "rank": i,
            "singular_value": float(S[i]),
            "variance_explained": float(S[i] ** 2 / total_var),
        })
    return {
        "layer": layer,
        "directions": directions,
        "top_1_variance": directions[0]["variance_explained"] if directions else 0,
    }


def mlp_output_token_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0, top_k: int = 5) -> dict:
    """How well MLP outputs align with unembed directions (token predictions).

    Shows which vocabulary tokens the MLP output points toward.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Project MLP output at last position to vocab space
    logits = mlp_out[-1] @ W_U  # [d_vocab]
    top_indices = jnp.argsort(logits)[::-1][:top_k]

    top_tokens = []
    for idx in top_indices:
        top_tokens.append({
            "token_id": int(idx),
            "logit": float(logits[idx]),
        })
    return {
        "layer": layer,
        "top_tokens": top_tokens,
        "logit_range": float(jnp.max(logits) - jnp.min(logits)),
        "mean_logit": float(jnp.mean(logits)),
    }


def mlp_output_position_variation(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0) -> dict:
    """How much MLP output varies across positions.

    High variation means the MLP is position-sensitive;
    low variation suggests position-independent processing.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    seq_len = mlp_out.shape[0]

    norms = jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = mlp_out / norms
    sim = normed @ normed.T
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    # Norm variation
    norm_vals = jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1))
    norm_cv = float(jnp.std(norm_vals) / jnp.mean(norm_vals).clip(1e-8))

    return {
        "layer": layer,
        "mean_position_similarity": mean_sim,
        "norm_coefficient_of_variation": norm_cv,
        "is_position_sensitive": mean_sim < 0.5,
    }


def mlp_output_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of MLP output space."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        rank = mlp_output_rank(model, tokens, layer)
        var = mlp_output_position_variation(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "effective_rank": rank["effective_rank"],
            "dim_for_90_pct": rank["dim_for_90_pct"],
            "position_similarity": var["mean_position_similarity"],
            "is_position_sensitive": var["is_position_sensitive"],
        })
    return {"per_layer": per_layer}
