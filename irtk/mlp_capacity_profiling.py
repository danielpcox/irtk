"""MLP capacity profiling: measure and analyze MLP layer capacity.

Profile how MLPs use their capacity: dead neurons, activation diversity,
weight utilization, and information throughput.
"""

import jax
import jax.numpy as jnp


def mlp_dead_neuron_profile(model, tokens, layer=0, threshold=0.01):
    """Identify dead and near-dead MLP neurons.

    A neuron is dead if its mean absolute activation is below threshold
    relative to the layer mean.

    Returns:
        dict with 'n_dead', 'dead_fraction', 'dead_neuron_ids',
        'per_neuron_activity'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    activity = jnp.mean(jnp.abs(hidden), axis=0)  # [d_mlp]
    mean_activity = float(jnp.mean(activity))
    abs_threshold = mean_activity * threshold
    dead = activity < abs_threshold
    dead_ids = [int(i) for i in range(hidden.shape[-1]) if bool(dead[i])]
    return {
        "n_dead": len(dead_ids),
        "dead_fraction": len(dead_ids) / hidden.shape[-1],
        "dead_neuron_ids": dead_ids[:20],  # cap for readability
        "n_total": int(hidden.shape[-1]),
        "mean_activity": mean_activity,
    }


def mlp_activation_diversity(model, tokens, layer=0):
    """How diverse are MLP activations across positions?

    Measures entropy of activation distribution per neuron.

    Returns:
        dict with 'mean_diversity', 'per_neuron_entropy' (top/bottom),
        'diversity_score'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    # Normalize activations to pseudo-probabilities per neuron
    abs_act = jnp.abs(hidden)  # [seq, d_mlp]
    sums = jnp.sum(abs_act, axis=0, keepdims=True) + 1e-10  # [1, d_mlp]
    probs = abs_act / sums  # [seq, d_mlp]
    # Entropy per neuron
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=0)  # [d_mlp]
    max_entropy = float(jnp.log(jnp.array(hidden.shape[0], dtype=jnp.float32)))
    normalized = entropy / (max_entropy + 1e-10)
    mean_div = float(jnp.mean(normalized))
    # Top and bottom neurons
    top_idx = jnp.argsort(-normalized)[:5]
    bottom_idx = jnp.argsort(normalized)[:5]
    return {
        "mean_diversity": mean_div,
        "most_diverse": [(int(i), float(normalized[i])) for i in top_idx],
        "least_diverse": [(int(i), float(normalized[i])) for i in bottom_idx],
        "diversity_score": mean_div,
    }


def mlp_weight_utilization(model, layer=0):
    """How much of the MLP weight capacity is being used?

    Measures via effective rank and singular value distribution.

    Returns:
        dict with 'w_in_effective_rank', 'w_out_effective_rank',
        'w_in_utilization', 'w_out_utilization'.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    def _effective_rank(W):
        s = jnp.linalg.svd(W, compute_uv=False)
        s = s / (jnp.sum(s) + 1e-10)
        s = jnp.where(s > 1e-10, s, 1e-10)
        return float(jnp.exp(-jnp.sum(s * jnp.log(s))))

    in_rank = _effective_rank(W_in)
    out_rank = _effective_rank(W_out)
    d_model = W_in.shape[0]
    d_mlp = W_in.shape[1]
    max_rank = min(d_model, d_mlp)
    return {
        "w_in_effective_rank": in_rank,
        "w_out_effective_rank": out_rank,
        "w_in_utilization": in_rank / max_rank,
        "w_out_utilization": out_rank / max_rank,
        "d_model": int(d_model),
        "d_mlp": int(d_mlp),
    }


def mlp_information_throughput(model, tokens, layer=0):
    """How much information passes through the MLP?

    Measured as the ratio of MLP output norm to input norm,
    and the fraction of residual explained by MLP.

    Returns:
        dict with 'mean_throughput_ratio', 'mean_residual_fraction',
        'per_position'.
    """
    _, cache = model.run_with_cache(tokens)
    resid_mid = cache[("resid_mid", layer)]  # [seq, d_model]
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    resid_post = cache[("resid_post", layer)]  # [seq, d_model]
    seq_len = resid_mid.shape[0]
    per_position = []
    for pos in range(seq_len):
        in_norm = float(jnp.linalg.norm(resid_mid[pos]))
        out_norm = float(jnp.linalg.norm(mlp_out[pos]))
        post_norm = float(jnp.linalg.norm(resid_post[pos]))
        ratio = out_norm / (in_norm + 1e-10)
        frac = out_norm / (post_norm + 1e-10)
        per_position.append({
            "position": pos,
            "throughput_ratio": ratio,
            "residual_fraction": frac,
        })
    mean_throughput = sum(p["throughput_ratio"] for p in per_position) / len(per_position)
    mean_frac = sum(p["residual_fraction"] for p in per_position) / len(per_position)
    return {
        "mean_throughput_ratio": mean_throughput,
        "mean_residual_fraction": mean_frac,
        "per_position": per_position,
    }


def mlp_capacity_summary(model, tokens):
    """Summary of MLP capacity across all layers.

    Returns:
        dict with 'per_layer' list of capacity metrics.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        dead = mlp_dead_neuron_profile(model, tokens, layer=layer)
        div = mlp_activation_diversity(model, tokens, layer=layer)
        util = mlp_weight_utilization(model, layer=layer)
        per_layer.append({
            "layer": layer,
            "dead_fraction": dead["dead_fraction"],
            "diversity_score": div["diversity_score"],
            "w_in_utilization": util["w_in_utilization"],
            "w_out_utilization": util["w_out_utilization"],
        })
    return {"per_layer": per_layer}
