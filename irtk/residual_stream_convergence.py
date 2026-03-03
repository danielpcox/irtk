"""Residual stream convergence: whether representations converge across layers."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def layer_to_layer_convergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track how much the residual changes between adjacent layers.

    Decreasing changes indicate convergence.
    """
    _, cache = model.run_with_cache(tokens)
    changes = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)]
        post = cache[("resid_post", layer)]
        delta = post - pre
        delta_norm = float(jnp.mean(jnp.sqrt(jnp.sum(delta ** 2, axis=-1))))
        pre_norm = float(jnp.mean(jnp.sqrt(jnp.sum(pre ** 2, axis=-1))).clip(1e-8))
        changes.append({
            "layer": layer,
            "absolute_change": delta_norm,
            "relative_change": delta_norm / pre_norm,
        })
    rel_changes = [c["relative_change"] for c in changes]
    is_converging = len(rel_changes) > 1 and rel_changes[-1] < rel_changes[0] * 0.8
    return {
        "per_layer": changes,
        "is_converging": is_converging,
        "convergence_ratio": rel_changes[-1] / max(rel_changes[0], 1e-8) if rel_changes else 1.0,
    }


def final_representation_stability(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Measure cosine similarity between each layer's output and the final layer.

    High similarity to the final layer = representation has stabilized.
    """
    _, cache = model.run_with_cache(tokens)
    final = cache[("resid_post", model.cfg.n_layers - 1)]  # [seq, d_model]
    final_norms = jnp.sqrt(jnp.sum(final ** 2, axis=-1, keepdims=True)).clip(1e-8)
    final_normed = final / final_norms

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        r_norms = jnp.sqrt(jnp.sum(resid ** 2, axis=-1, keepdims=True)).clip(1e-8)
        r_normed = resid / r_norms
        cos = jnp.mean(jnp.sum(r_normed * final_normed, axis=-1))
        per_layer.append({
            "layer": layer,
            "cosine_to_final": float(cos),
        })
    # Find stabilization layer: first layer with cosine > 0.95
    stable_layer = model.cfg.n_layers - 1
    for p in per_layer:
        if p["cosine_to_final"] > 0.95:
            stable_layer = p["layer"]
            break
    return {
        "per_layer": per_layer,
        "stabilization_layer": stable_layer,
    }


def position_convergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Do different positions converge to similar representations?

    Increasing mean similarity across layers indicates convergence.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        seq_len = resid.shape[0]
        norms = jnp.sqrt(jnp.sum(resid ** 2, axis=-1, keepdims=True)).clip(1e-8)
        normed = resid / norms
        sim = normed @ normed.T
        mask = 1.0 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))
        per_layer.append({
            "layer": layer,
            "mean_pairwise_similarity": mean_sim,
        })
    sims = [p["mean_pairwise_similarity"] for p in per_layer]
    trend = "converging" if sims[-1] > sims[0] + 0.1 else \
            "diverging" if sims[-1] < sims[0] - 0.1 else "stable"
    return {
        "per_layer": per_layer,
        "position_trend": trend,
    }


def norm_convergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track residual stream norm growth across layers.

    Bounded norms indicate the model is converging; explosive growth
    suggests instability.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        mean_norm = float(jnp.mean(jnp.sqrt(jnp.sum(resid ** 2, axis=-1))))
        std_norm = float(jnp.std(jnp.sqrt(jnp.sum(resid ** 2, axis=-1))))
        per_layer.append({
            "layer": layer,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
        })
    norms = [p["mean_norm"] for p in per_layer]
    growth = norms[-1] / max(norms[0], 1e-8)
    return {
        "per_layer": per_layer,
        "norm_growth_factor": growth,
        "is_stable": 0.5 < growth < 5.0,
    }


def convergence_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Combined convergence analysis."""
    l2l = layer_to_layer_convergence(model, tokens)
    stability = final_representation_stability(model, tokens)
    norms = norm_convergence(model, tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        per_layer.append({
            "layer": layer,
            "relative_change": l2l["per_layer"][layer]["relative_change"],
            "cosine_to_final": stability["per_layer"][layer]["cosine_to_final"],
            "mean_norm": norms["per_layer"][layer]["mean_norm"],
        })
    return {
        "per_layer": per_layer,
        "is_converging": l2l["is_converging"],
        "stabilization_layer": stability["stabilization_layer"],
        "norm_growth": norms["norm_growth_factor"],
        "is_stable": norms["is_stable"],
    }
