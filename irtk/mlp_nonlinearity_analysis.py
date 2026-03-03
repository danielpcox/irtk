"""MLP nonlinearity analysis: pre/post activation relationship and gating."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def activation_survival_rate(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Fraction of neurons that survive the nonlinearity (pre > 0 after ReLU).

    Low survival = heavy filtering; high survival = mostly pass-through.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[("pre", layer)]  # [seq, d_mlp]
    post = cache[("post", layer)]  # [seq, d_mlp]

    pre_positive = (pre > 0).astype(jnp.float32)
    post_active = (post > 0).astype(jnp.float32)

    per_position = []
    for pos in range(pre.shape[0]):
        pre_rate = float(jnp.mean(pre_positive[pos]))
        post_rate = float(jnp.mean(post_active[pos]))
        per_position.append({
            "position": pos,
            "pre_positive_rate": pre_rate,
            "post_active_rate": post_rate,
        })
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_survival_rate": float(jnp.mean(post_active)),
        "mean_pre_positive": float(jnp.mean(pre_positive)),
    }


def nonlinearity_distortion(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0) -> dict:
    """How much the nonlinearity distorts the activation pattern.

    Compares pre and post activation vectors via cosine similarity.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[("pre", layer)]
    post = cache[("post", layer)]

    per_position = []
    for pos in range(pre.shape[0]):
        p_norm = jnp.sqrt(jnp.sum(pre[pos] ** 2)).clip(1e-8)
        q_norm = jnp.sqrt(jnp.sum(post[pos] ** 2)).clip(1e-8)
        cos = float(jnp.sum(pre[pos] * post[pos]) / (p_norm * q_norm))
        norm_ratio = float(q_norm / p_norm)
        per_position.append({
            "position": pos,
            "cosine": cos,
            "norm_ratio": norm_ratio,
        })
    cosines = [p["cosine"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_cosine": sum(cosines) / len(cosines),
        "is_low_distortion": sum(cosines) / len(cosines) > 0.8,
    }


def activation_magnitude_shift(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0) -> dict:
    """Change in activation magnitudes through the nonlinearity.

    Shows how the nonlinearity scales different neurons.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[("pre", layer)]
    post = cache[("post", layer)]

    pre_mean = float(jnp.mean(jnp.abs(pre)))
    post_mean = float(jnp.mean(jnp.abs(post)))
    pre_max = float(jnp.max(jnp.abs(pre)))
    post_max = float(jnp.max(jnp.abs(post)))

    return {
        "layer": layer,
        "pre_mean_magnitude": pre_mean,
        "post_mean_magnitude": post_mean,
        "magnitude_ratio": post_mean / max(pre_mean, 1e-8),
        "pre_max": pre_max,
        "post_max": post_max,
    }


def neuron_selectivity_profile(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0, top_k: int = 10) -> dict:
    """Profile of the most selective neurons (active for fewest positions).

    Highly selective neurons may encode specific features.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]

    active_per_neuron = jnp.sum((post > 0).astype(jnp.float32), axis=0)  # [d_mlp]
    selectivity = 1.0 - active_per_neuron / post.shape[0]

    top_indices = jnp.argsort(-selectivity)[:top_k]
    top_neurons = []
    for idx in top_indices:
        idx_int = int(idx)
        top_neurons.append({
            "neuron": idx_int,
            "selectivity": float(selectivity[idx_int]),
            "active_positions": int(active_per_neuron[idx_int]),
        })

    return {
        "layer": layer,
        "top_selective": top_neurons,
        "mean_selectivity": float(jnp.mean(selectivity)),
        "n_always_active": int(jnp.sum(active_per_neuron == post.shape[0])),
        "n_never_active": int(jnp.sum(active_per_neuron == 0)),
    }


def mlp_nonlinearity_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer MLP nonlinearity summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        surv = activation_survival_rate(model, tokens, layer)
        dist = nonlinearity_distortion(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "survival_rate": surv["mean_survival_rate"],
            "distortion_cosine": dist["mean_cosine"],
            "is_low_distortion": dist["is_low_distortion"],
        })
    return {"per_layer": per_layer}
