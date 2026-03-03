"""Residual direction tracking: how directions evolve through layers."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def direction_persistence(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Track how much a direction persists from each layer to all later layers.

    High persistence = direction remains throughout processing.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    n_layers = model.cfg.n_layers
    resids = []
    for layer in range(n_layers):
        r = cache[("resid_post", layer)][position]
        norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        resids.append(r / norm)

    per_layer = []
    for i in range(n_layers):
        forward_cosines = []
        for j in range(i + 1, n_layers):
            cos = float(jnp.sum(resids[i] * resids[j]))
            forward_cosines.append(cos)
        per_layer.append({
            "layer": i,
            "mean_persistence": sum(forward_cosines) / max(len(forward_cosines), 1),
            "min_persistence": min(forward_cosines) if forward_cosines else 1.0,
        })

    return {
        "position": position,
        "per_layer": per_layer,
    }


def direction_change_rate(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Rate of direction change between consecutive layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    n_layers = model.cfg.n_layers
    per_transition = []
    for i in range(n_layers - 1):
        r1 = cache[("resid_post", i)][position]
        r2 = cache[("resid_post", i + 1)][position]
        n1 = jnp.sqrt(jnp.sum(r1 ** 2)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(r2 ** 2)).clip(1e-8)
        cos = float(jnp.sum(r1 * r2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        angle = float(jnp.arccos(jnp.array(cos)) * 180 / jnp.pi)
        per_transition.append({
            "layers": (i, i + 1),
            "cosine": cos,
            "angle_degrees": angle,
        })

    angles = [t["angle_degrees"] for t in per_transition]
    return {
        "position": position,
        "per_transition": per_transition,
        "mean_angle": sum(angles) / max(len(angles), 1),
        "max_angle": max(angles) if angles else 0,
    }


def unembed_direction_trajectory(model: HookedTransformer, tokens: jnp.ndarray,
                                    position: int = -1, token_id: int = 0) -> dict:
    """Track alignment with a specific unembedding direction through layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    target_dir = W_U[:, token_id]
    target_norm = jnp.sqrt(jnp.sum(target_dir ** 2)).clip(1e-8)
    target_dir = target_dir / target_norm

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        resid_norm = jnp.sqrt(jnp.sum(resid ** 2)).clip(1e-8)
        cos = float(jnp.sum(resid * target_dir) / resid_norm)
        projection = float(jnp.sum(resid * target_dir))
        per_layer.append({
            "layer": layer,
            "cosine": cos,
            "projection": projection,
        })

    return {
        "position": position,
        "token_id": token_id,
        "per_layer": per_layer,
    }


def dominant_direction_evolution(model: HookedTransformer, tokens: jnp.ndarray,
                                    position: int = -1) -> dict:
    """Track the dominant direction (top predicted token) through layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_top = None
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        top_token = int(jnp.argmax(logits))
        top_logit = float(jnp.max(logits))
        changed = prev_top is not None and top_token != prev_top
        per_layer.append({
            "layer": layer,
            "top_token": top_token,
            "top_logit": top_logit,
            "changed": changed,
        })
        prev_top = top_token

    n_changes = sum(1 for p in per_layer if p["changed"])
    return {
        "position": position,
        "per_layer": per_layer,
        "n_changes": n_changes,
        "final_token": per_layer[-1]["top_token"] if per_layer else None,
    }


def residual_direction_tracking_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                           position: int = -1) -> dict:
    """Combined direction tracking summary."""
    pers = direction_persistence(model, tokens, position)
    rate = direction_change_rate(model, tokens, position)
    dom = dominant_direction_evolution(model, tokens, position)
    return {
        "position": pers["position"],
        "mean_change_angle": rate["mean_angle"],
        "max_change_angle": rate["max_angle"],
        "n_prediction_changes": dom["n_changes"],
        "final_token": dom["final_token"],
    }
