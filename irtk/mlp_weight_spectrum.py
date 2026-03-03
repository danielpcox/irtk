"""MLP weight spectrum: spectral analysis of MLP weight matrices."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def mlp_input_weight_spectrum(model: HookedTransformer, layer: int = 0,
                                 top_k: int = 10) -> dict:
    """Singular value spectrum of W_in (input weight matrix)."""
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    svs = jnp.linalg.svd(W_in, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    return {
        "layer": layer,
        "top_singular_values": [float(s) for s in svs[:top_k]],
        "effective_rank": eff_rank,
        "condition_number": float(svs[0] / svs[-1].clip(1e-8)),
        "total_energy": float(jnp.sum(svs ** 2)),
    }


def mlp_output_weight_spectrum(model: HookedTransformer, layer: int = 0,
                                  top_k: int = 10) -> dict:
    """Singular value spectrum of W_out (output weight matrix)."""
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    svs = jnp.linalg.svd(W_out, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    return {
        "layer": layer,
        "top_singular_values": [float(s) for s in svs[:top_k]],
        "effective_rank": eff_rank,
        "condition_number": float(svs[0] / svs[-1].clip(1e-8)),
        "total_energy": float(jnp.sum(svs ** 2)),
    }


def mlp_in_out_alignment(model: HookedTransformer, layer: int = 0) -> dict:
    """Alignment between W_in and W_out via their shared structure."""
    W_in = model.blocks[layer].mlp.W_in   # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    # Product W_in @ W_out would be [d_model, d_model] — the residual-to-residual mapping
    product = W_in @ W_out  # [d_model, d_model]
    svs = jnp.linalg.svd(product, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    trace = float(jnp.trace(product))
    frob = float(jnp.sqrt(jnp.sum(product ** 2)))

    return {
        "layer": layer,
        "product_effective_rank": eff_rank,
        "product_trace": trace,
        "product_frobenius": frob,
        "top_svs": [float(s) for s in svs[:5]],
    }


def mlp_neuron_norm_distribution(model: HookedTransformer, layer: int = 0) -> dict:
    """Distribution of neuron norms (W_in columns and W_out rows)."""
    W_in = model.blocks[layer].mlp.W_in   # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    in_norms = jnp.sqrt(jnp.sum(W_in ** 2, axis=0))   # [d_mlp]
    out_norms = jnp.sqrt(jnp.sum(W_out ** 2, axis=1))  # [d_mlp]

    return {
        "layer": layer,
        "in_mean_norm": float(jnp.mean(in_norms)),
        "in_std_norm": float(jnp.std(in_norms)),
        "out_mean_norm": float(jnp.mean(out_norms)),
        "out_std_norm": float(jnp.std(out_norms)),
        "in_out_correlation": float(jnp.corrcoef(in_norms, out_norms)[0, 1]),
    }


def mlp_weight_spectrum_summary(model: HookedTransformer) -> dict:
    """Cross-layer MLP weight spectrum summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        w_in = mlp_input_weight_spectrum(model, layer)
        w_out = mlp_output_weight_spectrum(model, layer)
        per_layer.append({
            "layer": layer,
            "in_rank": w_in["effective_rank"],
            "out_rank": w_out["effective_rank"],
            "in_energy": w_in["total_energy"],
            "out_energy": w_out["total_energy"],
        })
    return {"per_layer": per_layer}
