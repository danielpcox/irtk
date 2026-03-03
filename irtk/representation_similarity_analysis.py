"""Representation similarity analysis: compare representations across layers and positions.

CKA (Centered Kernel Alignment), cosine similarity matrices, and
representation geometry for understanding how representations transform.
"""

import jax
import jax.numpy as jnp


def layer_representation_similarity(model, tokens, metric="cosine"):
    """Compare representations across all layers using cosine or CKA.

    Returns pairwise similarity matrix between layer representations.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    # Gather residual stream at each layer
    reps = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key in cache:
            reps.append(cache[key])
    if not reps:
        # Fallback to pre
        for layer in range(n_layers):
            key = f"blocks.{layer}.hook_resid_pre"
            if key in cache:
                reps.append(cache[key])

    n = len(reps)
    sim_matrix = jnp.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == "cosine":
                # Flatten to vectors and compute cosine
                a = reps[i].reshape(-1)
                b = reps[j].reshape(-1)
                cos = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10)
                sim_matrix = sim_matrix.at[i, j].set(cos)
            else:
                # CKA: centered kernel alignment
                X = reps[i]  # [seq, d_model]
                Y = reps[j]
                # Linear CKA
                XtX = X @ X.T
                YtY = Y @ Y.T
                XtY = X @ Y.T
                # Center
                n_s = X.shape[0]
                H = jnp.eye(n_s) - jnp.ones((n_s, n_s)) / n_s
                cXtX = H @ XtX @ H
                cYtY = H @ YtY @ H
                cXtY = H @ XtY @ H
                hsic_xy = jnp.sum(cXtY * cXtY.T)
                hsic_xx = jnp.sum(cXtX * cXtX.T)
                hsic_yy = jnp.sum(cYtY * cYtY.T)
                cka = hsic_xy / (jnp.sqrt(hsic_xx * hsic_yy) + 1e-10)
                sim_matrix = sim_matrix.at[i, j].set(cka)

    return {
        "similarity_matrix": sim_matrix,
        "metric": metric,
        "n_layers": n,
        "mean_adjacent_similarity": float(jnp.mean(jnp.array([
            sim_matrix[i, i + 1] for i in range(n - 1)
        ]))) if n > 1 else 0.0,
        "mean_distant_similarity": float(jnp.mean(jnp.array([
            sim_matrix[0, j] for j in range(2, n)
        ]))) if n > 2 else 0.0,
    }


def position_representation_similarity(model, tokens, layer=0):
    """Compare how similar different token positions are at a given layer.

    Returns pairwise cosine similarity between position representations.
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache:
        key = f"blocks.{layer}.hook_resid_pre"
    rep = cache[key]  # [seq, d_model]
    seq_len = rep.shape[0]

    # Normalize
    norms = jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-10
    normed = rep / norms

    sim_matrix = normed @ normed.T  # [seq, seq]

    # Diagonal is self-similarity (should be ~1)
    off_diag_mask = 1.0 - jnp.eye(seq_len)
    off_diag_vals = sim_matrix * off_diag_mask
    mean_off_diag = float(jnp.sum(off_diag_vals) / jnp.sum(off_diag_mask))

    # Adjacent similarity
    adj_sims = jnp.array([float(sim_matrix[i, i + 1]) for i in range(seq_len - 1)])
    mean_adjacent = float(jnp.mean(adj_sims)) if len(adj_sims) > 0 else 0.0

    return {
        "similarity_matrix": sim_matrix,
        "layer": layer,
        "seq_len": seq_len,
        "mean_pairwise_similarity": mean_off_diag,
        "mean_adjacent_similarity": mean_adjacent,
        "is_position_diverse": mean_off_diag < 0.5,
    }


def representation_drift(model, tokens, position=-1):
    """Track how a single position's representation changes across layers.

    Measures cosine distance from layer to layer and from layer to final.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        # Use last position
        position = len(tokens) - 1

    reps = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        reps.append(cache[key][position])

    final_rep = reps[-1]
    final_norm = jnp.linalg.norm(final_rep) + 1e-10

    per_layer = []
    for i, rep in enumerate(reps):
        norm = jnp.linalg.norm(rep) + 1e-10
        cos_to_final = float(jnp.dot(rep, final_rep) / (norm * final_norm))
        cos_to_prev = 0.0
        if i > 0:
            prev_norm = jnp.linalg.norm(reps[i - 1]) + 1e-10
            cos_to_prev = float(jnp.dot(rep, reps[i - 1]) / (norm * prev_norm))
        per_layer.append({
            "layer": i,
            "norm": float(norm),
            "cosine_to_final": cos_to_final,
            "cosine_to_previous": cos_to_prev,
        })

    total_drift = 1.0 - per_layer[0]["cosine_to_final"] if per_layer else 0.0

    return {
        "position": position,
        "per_layer": per_layer,
        "total_drift": total_drift,
        "n_layers": n_layers,
        "is_gradual": all(
            p["cosine_to_previous"] > 0.5 for p in per_layer[1:]
        ) if len(per_layer) > 1 else True,
    }


def representation_effective_dimension(model, tokens, layer=0):
    """Estimate the effective dimensionality of representations at a layer.

    Uses participation ratio of singular values.
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache:
        key = f"blocks.{layer}.hook_resid_pre"
    rep = cache[key]  # [seq, d_model]

    # Center
    rep_centered = rep - jnp.mean(rep, axis=0, keepdims=True)
    _, svals, _ = jnp.linalg.svd(rep_centered, full_matrices=False)

    # Participation ratio
    svals_sq = svals ** 2
    total = jnp.sum(svals_sq)
    participation_ratio = float(total ** 2 / (jnp.sum(svals_sq ** 2) + 1e-10))

    # Variance explained
    cumvar = jnp.cumsum(svals_sq) / (total + 1e-10)
    dim_90 = int(jnp.searchsorted(cumvar, 0.9)) + 1

    return {
        "layer": layer,
        "participation_ratio": participation_ratio,
        "dim_for_90_pct": dim_90,
        "top_singular_value": float(svals[0]),
        "singular_values": svals,
        "is_low_dimensional": participation_ratio < min(rep.shape) * 0.3,
    }


def representation_geometry_summary(model, tokens):
    """Cross-layer summary of representation geometry.

    Tracks effective dimension, isotropy, and norm statistics per layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key]  # [seq, d_model]

        # Norms
        norms = jnp.linalg.norm(rep, axis=-1)
        mean_norm = float(jnp.mean(norms))

        # Pairwise cosine
        normed = rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-10)
        sim = normed @ normed.T
        seq_len = rep.shape[0]
        off_diag_mask = 1.0 - jnp.eye(seq_len)
        mean_cos = float(jnp.sum(sim * off_diag_mask) / (jnp.sum(off_diag_mask) + 1e-10))

        # Quick effective dim via participation ratio
        rep_c = rep - jnp.mean(rep, axis=0, keepdims=True)
        _, sv, _ = jnp.linalg.svd(rep_c, full_matrices=False)
        sv_sq = sv ** 2
        total = jnp.sum(sv_sq) + 1e-10
        pr = float(total ** 2 / (jnp.sum(sv_sq ** 2) + 1e-10))

        per_layer.append({
            "layer": layer,
            "mean_norm": mean_norm,
            "mean_pairwise_cosine": mean_cos,
            "participation_ratio": pr,
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
        "geometry_trend": "expanding" if per_layer[-1]["participation_ratio"] > per_layer[0]["participation_ratio"] else "contracting",
    }
