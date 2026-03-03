"""MLP input-output mapping: linearity, selectivity, and transformation structure."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def mlp_linearity_measure(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0) -> dict:
    """How linear is the MLP's input-output mapping?

    Compares actual output to best linear approximation.
    High linearity = nonlinearity has little effect on directions.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]  # [seq, d_model]
    resid_mid = cache[("resid_mid", layer)]  # [seq, d_model] (after attn, before MLP)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]

    # Input to MLP is resid_mid
    # Best linear mapping: least squares (mlp_out = resid_mid @ W + b)
    # Use correlation as proxy for linearity
    per_position = []
    for pos in range(resid_mid.shape[0]):
        inp = resid_mid[pos]
        out = mlp_out[pos]
        i_norm = jnp.sqrt(jnp.sum(inp ** 2)).clip(1e-8)
        o_norm = jnp.sqrt(jnp.sum(out ** 2)).clip(1e-8)
        cos = float(jnp.sum(inp * out) / (i_norm * o_norm))
        per_position.append({
            "position": pos,
            "input_output_cosine": cos,
            "input_norm": float(i_norm),
            "output_norm": float(o_norm),
        })

    cosines = [p["input_output_cosine"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_cosine": sum(cosines) / len(cosines),
        "is_linear": abs(sum(cosines) / len(cosines)) > 0.7,
    }


def mlp_input_selectivity(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0) -> dict:
    """How selective is the MLP to different inputs?

    High variance in output norms = selective (different responses).
    Low variance = uniform processing.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]

    norms = jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1))  # [seq]
    mean_norm = float(jnp.mean(norms))
    std_norm = float(jnp.std(norms))
    cv = std_norm / max(mean_norm, 1e-8)  # coefficient of variation

    per_position = []
    for pos in range(mlp_out.shape[0]):
        per_position.append({
            "position": pos,
            "output_norm": float(norms[pos]),
        })

    return {
        "layer": layer,
        "per_position": per_position,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "coefficient_of_variation": cv,
        "is_selective": cv > 0.3,
    }


def mlp_transformation_structure(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0) -> dict:
    """Structure of the MLP transformation: amplification, rotation, etc.

    Analyzes how the MLP changes the residual stream direction and magnitude.
    """
    _, cache = model.run_with_cache(tokens)
    resid_mid = cache[("resid_mid", layer)]
    mlp_out = cache[("mlp_out", layer)]
    resid_post = cache[("resid_post", layer)]

    per_position = []
    for pos in range(resid_mid.shape[0]):
        pre = resid_mid[pos]
        out = mlp_out[pos]
        post = resid_post[pos]

        pre_norm = float(jnp.sqrt(jnp.sum(pre ** 2)).clip(1e-8))
        out_norm = float(jnp.sqrt(jnp.sum(out ** 2)))
        post_norm = float(jnp.sqrt(jnp.sum(post ** 2)))

        # Direction change
        pre_dir = pre / jnp.sqrt(jnp.sum(pre ** 2)).clip(1e-8)
        post_dir = post / jnp.sqrt(jnp.sum(post ** 2)).clip(1e-8)
        cos_change = float(jnp.sum(pre_dir * post_dir))

        per_position.append({
            "position": pos,
            "norm_change": post_norm / max(pre_norm, 1e-8),
            "direction_change_cosine": cos_change,
            "mlp_magnitude": out_norm,
        })

    cos_changes = [p["direction_change_cosine"] for p in per_position]
    norm_changes = [p["norm_change"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_direction_change": sum(cos_changes) / len(cos_changes),
        "mean_norm_change": sum(norm_changes) / len(norm_changes),
        "is_amplifying": sum(norm_changes) / len(norm_changes) > 1.1,
    }


def mlp_output_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                            layer: int = 0) -> dict:
    """How diverse are MLP outputs across positions?

    Low diversity = MLP writes similar things everywhere.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    seq_len = mlp_out.shape[0]

    norms = jnp.sqrt(jnp.sum(mlp_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = mlp_out / norms
    sim = normed @ normed.T
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    return {
        "layer": layer,
        "mean_output_similarity": mean_sim,
        "is_diverse": mean_sim < 0.5,
    }


def mlp_mapping_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer MLP mapping summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        lin = mlp_linearity_measure(model, tokens, layer)
        sel = mlp_input_selectivity(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_cosine": lin["mean_cosine"],
            "is_linear": lin["is_linear"],
            "selectivity_cv": sel["coefficient_of_variation"],
            "is_selective": sel["is_selective"],
        })
    return {"per_layer": per_layer}
