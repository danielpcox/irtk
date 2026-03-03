"""Residual stream geometry: geometric structure of the residual stream.

Analyze angles, distances, subspaces, and geometric transformations
applied by each layer to the residual stream.
"""

import jax
import jax.numpy as jnp


def residual_angle_structure(model, tokens, position=-1):
    """Measure angles between the residual stream and key directions across layers.

    Tracks alignment with embedding, unembedding, and previous layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_E = model.embed.W_E
    W_U = model.unembed.W_U
    token_id = int(tokens[position])

    embed_dir = W_E[token_id]
    embed_norm = jnp.linalg.norm(embed_dir) + 1e-10

    # Unembed direction for the most-predicted token at final layer
    final_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if final_key not in cache:
        final_key = f"blocks.{n_layers - 1}.hook_resid_pre"
    final_rep = cache[final_key][position]
    final_logits = final_rep @ W_U
    target_token = int(jnp.argmax(final_logits))
    unembed_dir = W_U[:, target_token]
    unembed_norm = jnp.linalg.norm(unembed_dir) + 1e-10

    per_layer = []
    prev_rep = None
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]
        rep_norm = jnp.linalg.norm(rep) + 1e-10

        angle_to_embed = float(jnp.dot(rep, embed_dir) / (rep_norm * embed_norm))
        angle_to_unembed = float(jnp.dot(rep, unembed_dir) / (rep_norm * unembed_norm))

        angle_to_prev = 0.0
        if prev_rep is not None:
            prev_norm = jnp.linalg.norm(prev_rep) + 1e-10
            angle_to_prev = float(jnp.dot(rep, prev_rep) / (rep_norm * prev_norm))

        per_layer.append({
            "layer": layer,
            "cosine_to_embed": angle_to_embed,
            "cosine_to_unembed": angle_to_unembed,
            "cosine_to_previous": angle_to_prev,
            "norm": float(rep_norm),
        })
        prev_rep = rep

    return {
        "per_layer": per_layer,
        "position": position,
        "target_token": target_token,
    }


def residual_subspace_dimension(model, tokens, layer=0):
    """Estimate the dimensionality of the subspace occupied by residual vectors.

    Uses SVD to find the effective rank of the representation matrix.
    """
    _, cache = model.run_with_cache(tokens)

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache:
        key = f"blocks.{layer}.hook_resid_pre"
    rep = cache[key]  # [seq, d_model]

    # Center
    rep_c = rep - jnp.mean(rep, axis=0, keepdims=True)
    _, svals, Vt = jnp.linalg.svd(rep_c, full_matrices=False)

    svals_sq = svals ** 2
    total = jnp.sum(svals_sq) + 1e-10

    # Participation ratio
    pr = float(total ** 2 / (jnp.sum(svals_sq ** 2) + 1e-10))

    # Variance explained
    cumvar = jnp.cumsum(svals_sq) / total
    dim_90 = int(jnp.searchsorted(cumvar, 0.9)) + 1
    dim_95 = int(jnp.searchsorted(cumvar, 0.95)) + 1

    return {
        "layer": layer,
        "participation_ratio": pr,
        "dim_for_90_pct": dim_90,
        "dim_for_95_pct": dim_95,
        "top_sv_fraction": float(svals_sq[0] / total),
        "singular_values": svals,
    }


def residual_update_geometry(model, tokens, position=-1):
    """Analyze the geometric relationship between residual updates and the stream.

    Decomposes updates into parallel and perpendicular components.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    per_layer = []
    for layer in range(n_layers):
        pre_key = f"blocks.{layer}.hook_resid_pre"
        post_key = f"blocks.{layer}.hook_resid_post"

        if pre_key not in cache or post_key not in cache:
            continue

        pre = cache[pre_key][position]
        post = cache[post_key][position]
        update = post - pre

        pre_norm = jnp.linalg.norm(pre) + 1e-10
        update_norm = jnp.linalg.norm(update) + 1e-10
        unit_pre = pre / pre_norm

        # Parallel component (along residual direction)
        parallel_mag = float(jnp.dot(update, unit_pre))
        parallel_comp = parallel_mag * unit_pre
        perp_comp = update - parallel_comp
        perp_mag = float(jnp.linalg.norm(perp_comp))

        per_layer.append({
            "layer": layer,
            "update_norm": float(update_norm),
            "parallel_magnitude": parallel_mag,
            "perpendicular_magnitude": perp_mag,
            "angle_to_residual": float(jnp.dot(update, unit_pre) / (update_norm + 1e-10)),
            "is_mostly_perpendicular": perp_mag > abs(parallel_mag),
        })

    return {
        "per_layer": per_layer,
        "position": position,
        "mostly_perpendicular": sum(1 for p in per_layer if p["is_mostly_perpendicular"]),
    }


def residual_pairwise_distances(model, tokens, layer=0):
    """Compute pairwise distances between token positions in residual space.

    Shows clustering and separation patterns.
    """
    _, cache = model.run_with_cache(tokens)

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache:
        key = f"blocks.{layer}.hook_resid_pre"
    rep = cache[key]  # [seq, d_model]
    seq_len = rep.shape[0]

    # Pairwise L2 distances
    diff = rep[:, None, :] - rep[None, :, :]  # [seq, seq, d_model]
    distances = jnp.linalg.norm(diff, axis=-1)  # [seq, seq]

    off_diag_mask = 1.0 - jnp.eye(seq_len)
    mean_dist = float(jnp.sum(distances * off_diag_mask) / (jnp.sum(off_diag_mask) + 1e-10))
    max_dist = float(jnp.max(distances * off_diag_mask))
    min_dist = float(jnp.min(distances + jnp.eye(seq_len) * 1e10))

    return {
        "distance_matrix": distances,
        "layer": layer,
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "min_distance": min_dist,
        "spread_ratio": max_dist / (min_dist + 1e-10),
    }


def residual_geometry_summary(model, tokens):
    """Cross-layer summary of residual stream geometry.

    Tracks norm growth, subspace dimension, and update patterns.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key]
        norms = jnp.linalg.norm(rep, axis=-1)
        mean_norm = float(jnp.mean(norms))

        # Quick subspace dim
        rep_c = rep - jnp.mean(rep, axis=0, keepdims=True)
        _, sv, _ = jnp.linalg.svd(rep_c, full_matrices=False)
        sv_sq = sv ** 2
        total = jnp.sum(sv_sq) + 1e-10
        pr = float(total ** 2 / (jnp.sum(sv_sq ** 2) + 1e-10))

        per_layer.append({
            "layer": layer,
            "mean_norm": mean_norm,
            "participation_ratio": pr,
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
        "norm_trend": "growing" if per_layer[-1]["mean_norm"] > per_layer[0]["mean_norm"] else "shrinking",
    }
