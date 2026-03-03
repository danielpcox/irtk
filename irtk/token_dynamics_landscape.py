"""Token dynamics landscape: map how token representations evolve through layers.

Visualize the trajectory of token representations through the model as a
landscape, tracking velocity, curvature, convergence, and divergence.
"""

import jax
import jax.numpy as jnp


def token_velocity(model, tokens):
    """How fast does each token's representation change per layer?

    Measures the L2 norm of the residual stream delta at each layer.

    Returns:
        dict with 'per_layer' list of per-position velocities,
        'mean_velocity_per_layer', 'fastest_position'.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        post = cache[("resid_post", layer)]  # [seq, d_model]
        if layer == 0:
            pre = cache[("resid_pre", 0)]
        else:
            pre = cache[("resid_post", layer - 1)]
        delta = post - pre  # [seq, d_model]
        norms = jnp.linalg.norm(delta, axis=-1)  # [seq]
        per_layer.append({
            "layer": layer,
            "per_position": [float(norms[p]) for p in range(len(tokens))],
            "mean_velocity": float(jnp.mean(norms)),
            "max_velocity": float(jnp.max(norms)),
        })
    mean_per_layer = [p["mean_velocity"] for p in per_layer]
    fastest_layer = int(jnp.argmax(jnp.array(mean_per_layer)))
    return {
        "per_layer": per_layer,
        "mean_velocity_per_layer": mean_per_layer,
        "fastest_layer": fastest_layer,
    }


def token_curvature(model, tokens):
    """How much does the direction of change shift between consecutive layers?

    Curvature = 1 - cosine(delta_l, delta_{l+1}).

    Returns:
        dict with 'per_transition' list, 'mean_curvature'.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    deltas = []
    for layer in range(n_layers):
        post = cache[("resid_post", layer)]
        if layer == 0:
            pre = cache[("resid_pre", 0)]
        else:
            pre = cache[("resid_post", layer - 1)]
        deltas.append(post - pre)  # [seq, d_model]
    per_transition = []
    for i in range(1, n_layers):
        d_prev = deltas[i - 1]  # [seq, d_model]
        d_curr = deltas[i]
        # Mean cosine across positions
        cos_per_pos = jnp.sum(d_prev * d_curr, axis=-1) / (
            jnp.linalg.norm(d_prev, axis=-1) * jnp.linalg.norm(d_curr, axis=-1) + 1e-10
        )
        mean_cos = float(jnp.mean(cos_per_pos))
        curvature = 1.0 - mean_cos
        per_transition.append({
            "from_layer": i - 1,
            "to_layer": i,
            "mean_cosine": mean_cos,
            "curvature": curvature,
        })
    mean_curv = sum(p["curvature"] for p in per_transition) / len(per_transition) if per_transition else 0.0
    return {
        "per_transition": per_transition,
        "mean_curvature": mean_curv,
    }


def token_convergence(model, tokens):
    """Do different token positions converge or diverge through layers?

    Measures mean pairwise cosine similarity of residual streams.

    Returns:
        dict with 'per_layer' similarity, 'trend' (converging/diverging/stable).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    seq_len = len(tokens)
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]  # [seq, d_model]
        norms = jnp.linalg.norm(resid, axis=-1, keepdims=True) + 1e-10
        normed = resid / norms
        sim = normed @ normed.T  # [seq, seq]
        mask = jnp.ones((seq_len, seq_len)) - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / (jnp.sum(mask) + 1e-10))
        per_layer.append({
            "layer": layer,
            "mean_pairwise_similarity": mean_sim,
        })
    sims = [p["mean_pairwise_similarity"] for p in per_layer]
    if len(sims) >= 2:
        diff = sims[-1] - sims[0]
        trend = "converging" if diff > 0.05 else ("diverging" if diff < -0.05 else "stable")
    else:
        trend = "stable"
    return {
        "per_layer": per_layer,
        "trend": trend,
    }


def token_representation_drift(model, tokens, reference_layer=0):
    """How far has each token drifted from its representation at reference_layer?

    Returns:
        dict with 'per_layer' list of per-position cosine drifts.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    ref = cache[("resid_post", reference_layer)]  # [seq, d_model]
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]
        # Per-position cosine
        cos = jnp.sum(ref * resid, axis=-1) / (
            jnp.linalg.norm(ref, axis=-1) * jnp.linalg.norm(resid, axis=-1) + 1e-10
        )
        drift = 1.0 - cos  # 0 = same direction, 2 = opposite
        per_layer.append({
            "layer": layer,
            "per_position_drift": [float(drift[p]) for p in range(len(tokens))],
            "mean_drift": float(jnp.mean(drift)),
        })
    return {
        "reference_layer": reference_layer,
        "per_layer": per_layer,
    }


def token_dynamics_landscape_summary(model, tokens):
    """Summary of token dynamics through the model.

    Returns:
        dict with 'fastest_layer', 'mean_curvature', 'convergence_trend',
        'per_layer' velocities.
    """
    vel = token_velocity(model, tokens)
    curv = token_curvature(model, tokens)
    conv = token_convergence(model, tokens)
    return {
        "fastest_layer": vel["fastest_layer"],
        "mean_curvature": curv["mean_curvature"],
        "convergence_trend": conv["trend"],
        "per_layer": vel["per_layer"],
    }
