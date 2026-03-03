"""Token representation geometry: manifold structure and clustering."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def representation_clustering(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0) -> dict:
    """How tightly clustered are token representations at a given layer?

    High mean similarity = tokens are similar (collapsed); low = diverse.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    seq_len = resid.shape[0]

    norms = jnp.sqrt(jnp.sum(resid ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = resid / norms
    sim = normed @ normed.T
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    per_position = []
    for pos in range(seq_len):
        pos_sims = []
        for other in range(seq_len):
            if other != pos:
                pos_sims.append(float(sim[pos, other]))
        mean_pos_sim = sum(pos_sims) / max(len(pos_sims), 1)
        per_position.append({
            "position": pos,
            "mean_similarity": mean_pos_sim,
            "norm": float(norms[pos, 0]),
        })

    return {
        "layer": layer,
        "per_position": per_position,
        "mean_pairwise_similarity": mean_sim,
        "is_clustered": mean_sim > 0.5,
    }


def representation_spread(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0) -> dict:
    """Spread of representations: variance and effective dimensionality.

    Uses SVD on the centered representations.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]

    # Center the representations
    centered = resid - jnp.mean(resid, axis=0, keepdims=True)
    svs = jnp.linalg.svd(centered, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    total_var = float(jnp.sum(svs ** 2))
    top_sv_fraction = float(svs[0] ** 2 / max(total_var, 1e-8))

    return {
        "layer": layer,
        "effective_rank": eff_rank,
        "total_variance": total_var,
        "top_sv_fraction": top_sv_fraction,
        "is_low_rank": top_sv_fraction > 0.5,
    }


def representation_velocity(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """How fast the representation changes between layers.

    Analogous to velocity in representation space.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    per_layer = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)][position]
        post = cache[("resid_post", layer)][position]
        update = post - pre
        velocity = float(jnp.sqrt(jnp.sum(update ** 2)))
        pre_norm = float(jnp.sqrt(jnp.sum(pre ** 2)).clip(1e-8))
        per_layer.append({
            "layer": layer,
            "velocity": velocity,
            "relative_velocity": velocity / pre_norm,
        })

    velocities = [p["velocity"] for p in per_layer]
    # Acceleration = change in velocity
    accelerations = []
    for i in range(len(velocities) - 1):
        accelerations.append(velocities[i + 1] - velocities[i])

    return {
        "position": position,
        "per_layer": per_layer,
        "mean_velocity": sum(velocities) / len(velocities),
        "is_decelerating": len(accelerations) > 0 and accelerations[-1] < 0,
    }


def inter_token_distances(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0) -> dict:
    """Pairwise Euclidean distances between token representations.

    Shows which tokens are close (similar) vs far (different).
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    seq_len = resid.shape[0]

    distances = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            dist = float(jnp.sqrt(jnp.sum((resid[i] - resid[j]) ** 2)))
            distances.append({
                "positions": (i, j),
                "distance": dist,
            })

    dists = [d["distance"] for d in distances]
    return {
        "layer": layer,
        "distances": distances,
        "mean_distance": sum(dists) / max(len(dists), 1),
        "min_distance": min(dists) if dists else 0,
        "max_distance": max(dists) if dists else 0,
    }


def representation_geometry_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer representation geometry summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        cluster = representation_clustering(model, tokens, layer)
        spread = representation_spread(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_similarity": cluster["mean_pairwise_similarity"],
            "effective_rank": spread["effective_rank"],
            "is_clustered": cluster["is_clustered"],
        })
    return {"per_layer": per_layer}
