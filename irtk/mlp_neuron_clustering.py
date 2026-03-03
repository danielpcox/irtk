"""MLP neuron clustering: grouping neurons by behavior and activation patterns."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def neuron_activation_similarity(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0) -> dict:
    """Pairwise cosine similarity between neuron activation patterns.

    Each neuron's activation across positions forms a vector;
    similar vectors mean neurons activate on the same positions.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]
    d_mlp = post.shape[1]

    # Neuron activations: each column is a neuron's activation profile
    norms = jnp.sqrt(jnp.sum(post ** 2, axis=0, keepdims=True)).clip(1e-8)  # [1, d_mlp]
    normed = post / norms  # [seq, d_mlp]
    sim = normed.T @ normed  # [d_mlp, d_mlp]

    mask = 1.0 - jnp.eye(d_mlp)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))
    max_sim = float(jnp.max(sim * mask))

    return {
        "layer": layer,
        "n_neurons": int(d_mlp),
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "is_diverse": mean_sim < 0.3,
    }


def neuron_activity_profile(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0, top_k: int = 10) -> dict:
    """Profile neuron activity: which are most/least active.

    Returns activation statistics and top active neurons.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]

    # Mean activation per neuron
    mean_act = jnp.mean(post, axis=0)  # [d_mlp]
    max_act = jnp.max(post, axis=0)  # [d_mlp]

    # Sparsity: fraction of positions where neuron is active (>0)
    active_count = jnp.sum(post > 0, axis=0)  # [d_mlp]
    sparsity = 1.0 - active_count / post.shape[0]

    # Top-k most active by mean
    top_indices = jnp.argsort(mean_act)[::-1][:top_k]
    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append({
            "neuron": idx,
            "mean_activation": float(mean_act[idx]),
            "max_activation": float(max_act[idx]),
            "sparsity": float(sparsity[idx]),
        })

    return {
        "layer": layer,
        "top_active": top_neurons,
        "mean_sparsity": float(jnp.mean(sparsity)),
        "n_dead": int(jnp.sum(max_act <= 0)),
    }


def neuron_coactivation(model: HookedTransformer, tokens: jnp.ndarray,
                         layer: int = 0, top_k: int = 5) -> dict:
    """Find pairs of neurons that frequently co-activate.

    Co-activation suggests neurons form functional groups.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]
    d_mlp = post.shape[1]

    # Binary activation matrix
    active = (post > 0).astype(jnp.float32)  # [seq, d_mlp]
    coact = active.T @ active  # [d_mlp, d_mlp]
    coact_normed = coact / post.shape[0]

    # Mask diagonal
    mask = 1.0 - jnp.eye(d_mlp)
    coact_masked = coact_normed * mask

    # Top co-activating pairs
    flat = coact_masked.flatten()
    top_flat = jnp.argsort(flat)[::-1][:top_k]
    pairs = []
    for idx in top_flat:
        i = int(idx) // d_mlp
        j = int(idx) % d_mlp
        pairs.append({
            "neuron_a": i,
            "neuron_b": j,
            "coactivation_rate": float(coact_normed[i, j]),
        })

    mean_coact = float(jnp.sum(coact_masked) / jnp.sum(mask).clip(1e-8))
    return {
        "layer": layer,
        "top_pairs": pairs,
        "mean_coactivation": mean_coact,
    }


def neuron_output_direction_clustering(model: HookedTransformer, layer: int = 0,
                                        top_k: int = 5) -> dict:
    """Cluster neurons by their output direction in W_out.

    Neurons with similar output directions write similar things
    to the residual stream.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = W_out.shape[0]

    norms = jnp.sqrt(jnp.sum(W_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = W_out / norms  # [d_mlp, d_model]
    sim = normed @ normed.T  # [d_mlp, d_mlp]

    mask = 1.0 - jnp.eye(d_mlp)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    # Top similar pairs
    flat = (sim * mask).flatten()
    top_flat = jnp.argsort(flat)[::-1][:top_k]
    pairs = []
    for idx in top_flat:
        i = int(idx) // d_mlp
        j = int(idx) % d_mlp
        pairs.append({
            "neuron_a": i,
            "neuron_b": j,
            "direction_similarity": float(sim[i, j]),
        })

    return {
        "layer": layer,
        "mean_direction_similarity": mean_sim,
        "most_similar_pairs": pairs,
        "is_clustered": mean_sim > 0.3,
    }


def neuron_clustering_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of neuron clustering patterns."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        act_sim = neuron_activation_similarity(model, tokens, layer)
        profile = neuron_activity_profile(model, tokens, layer, top_k=3)
        per_layer.append({
            "layer": layer,
            "activation_similarity": act_sim["mean_similarity"],
            "mean_sparsity": profile["mean_sparsity"],
            "n_dead": profile["n_dead"],
            "is_diverse": act_sim["is_diverse"],
        })
    return {"per_layer": per_layer}
