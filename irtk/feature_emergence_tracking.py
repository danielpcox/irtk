"""Feature emergence tracking: track when specific features emerge across layers.

Detect which layers introduce new features into the residual stream,
measure feature strength evolution, and find critical emergence points.
"""

import jax
import jax.numpy as jnp


def feature_probe_trajectory(model, tokens, direction, position=-1):
    """Track how strongly a specific direction is represented across layers.

    Projects the residual stream onto a given direction at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    direction_norm = jnp.linalg.norm(direction) + 1e-10
    unit_dir = direction / direction_norm

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]  # [d_model]
        projection = float(jnp.dot(rep, unit_dir))
        rep_norm = float(jnp.linalg.norm(rep))
        per_layer.append({
            "layer": layer,
            "projection": projection,
            "abs_projection": abs(projection),
            "fraction_of_norm": abs(projection) / (rep_norm + 1e-10),
        })

    # Find emergence point: first layer where projection exceeds threshold
    max_proj = max(p["abs_projection"] for p in per_layer) if per_layer else 0.0
    emergence_layer = 0
    for p in per_layer:
        if p["abs_projection"] > 0.5 * max_proj:
            emergence_layer = p["layer"]
            break

    return {
        "per_layer": per_layer,
        "emergence_layer": emergence_layer,
        "max_projection": max_proj,
        "position": position,
    }


def token_feature_emergence(model, tokens, position=-1, top_k=5):
    """Track which vocabulary tokens become most predicted across layers.

    Uses logit lens to see the evolving prediction at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U  # [d_model, vocab]
    b_U = model.unembed.b_U  # [vocab]

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]  # [d_model]
        logits = rep @ W_U + b_U  # [vocab]
        probs = jax.nn.softmax(logits)

        top_indices = jnp.argsort(logits)[-top_k:][::-1]
        top_tokens = []
        for idx in top_indices:
            top_tokens.append({
                "token_id": int(idx),
                "logit": float(logits[idx]),
                "probability": float(probs[idx]),
            })

        max_prob = float(jnp.max(probs))
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))

        per_layer.append({
            "layer": layer,
            "top_token": int(top_indices[0]),
            "top_prob": max_prob,
            "entropy": entropy,
            "top_tokens": top_tokens,
        })

    # Find prediction commit point
    final_pred = per_layer[-1]["top_token"] if per_layer else -1
    commit_layer = n_layers - 1
    for p in per_layer:
        if p["top_token"] == final_pred:
            commit_layer = p["layer"]
            break

    return {
        "per_layer": per_layer,
        "final_prediction": final_pred,
        "commit_layer": commit_layer,
        "position": position,
    }


def component_feature_contribution(model, tokens, direction, position=-1):
    """Decompose feature emergence into attention vs MLP contributions.

    For a given direction, measures how much each component pushes
    the representation toward that direction.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    direction_norm = jnp.linalg.norm(direction) + 1e-10
    unit_dir = direction / direction_norm

    per_layer = []
    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        attn_proj = 0.0
        mlp_proj = 0.0
        if attn_key in cache:
            attn_out = cache[attn_key][position]
            attn_proj = float(jnp.dot(attn_out, unit_dir))
        if mlp_key in cache:
            mlp_out = cache[mlp_key][position]
            mlp_proj = float(jnp.dot(mlp_out, unit_dir))

        per_layer.append({
            "layer": layer,
            "attn_contribution": attn_proj,
            "mlp_contribution": mlp_proj,
            "total_contribution": attn_proj + mlp_proj,
            "dominant": "attention" if abs(attn_proj) > abs(mlp_proj) else "mlp",
        })

    return {
        "per_layer": per_layer,
        "total_attn": sum(p["attn_contribution"] for p in per_layer),
        "total_mlp": sum(p["mlp_contribution"] for p in per_layer),
        "position": position,
    }


def feature_interference(model, tokens, directions, position=-1):
    """Measure interference between multiple feature directions across layers.

    Checks if features compete for representation capacity or coexist.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    n_dirs = len(directions)
    # Normalize directions
    unit_dirs = []
    for d in directions:
        norm = jnp.linalg.norm(d) + 1e-10
        unit_dirs.append(d / norm)

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]

        projections = [float(jnp.dot(rep, ud)) for ud in unit_dirs]
        # Pairwise interference: cosine between directions
        pairs = []
        for i in range(n_dirs):
            for j in range(i + 1, n_dirs):
                cos = float(jnp.dot(unit_dirs[i], unit_dirs[j]))
                pairs.append({
                    "dir_i": i, "dir_j": j,
                    "cosine": cos,
                })

        per_layer.append({
            "layer": layer,
            "projections": projections,
            "direction_pairs": pairs,
        })

    # Mean interference across direction pairs
    if per_layer and per_layer[0]["direction_pairs"]:
        mean_interference = sum(
            abs(p["cosine"]) for pair in per_layer for p in pair["direction_pairs"]
        ) / (len(per_layer) * len(per_layer[0]["direction_pairs"]))
    else:
        mean_interference = 0.0

    return {
        "per_layer": per_layer,
        "n_directions": n_dirs,
        "mean_interference": mean_interference,
        "high_interference": mean_interference > 0.5,
    }


def feature_emergence_summary(model, tokens, position=-1, top_k=5):
    """Summary of feature emergence across layers.

    Tracks prediction stability, entropy reduction, and new feature introduction.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_rep = None
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]

        # Logit lens
        logits = rep @ W_U + b_U
        probs = jax.nn.softmax(logits)
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
        top_token = int(jnp.argmax(logits))
        max_prob = float(jnp.max(probs))

        # New information (perpendicular component)
        new_info_frac = 0.0
        if prev_rep is not None:
            update = rep - prev_rep
            update_norm = jnp.linalg.norm(update) + 1e-10
            rep_norm = jnp.linalg.norm(rep) + 1e-10
            new_info_frac = float(update_norm / rep_norm)

        per_layer.append({
            "layer": layer,
            "top_token": top_token,
            "max_prob": max_prob,
            "entropy": entropy,
            "new_info_fraction": new_info_frac,
        })
        prev_rep = rep

    return {
        "per_layer": per_layer,
        "entropy_reduction": per_layer[0]["entropy"] - per_layer[-1]["entropy"] if per_layer else 0.0,
        "position": position,
    }
