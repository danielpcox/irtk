"""MLP bottleneck analysis: compression and expansion through MLPs.

Analyze how information is compressed into the MLP hidden layer
and expanded back, including effective dimensionality and utilization.
"""

import jax
import jax.numpy as jnp


def mlp_compression_ratio(model, tokens, layer=0):
    """Effective compression ratio of the MLP bottleneck.

    Compares the effective rank of the input vs hidden representations.

    Returns:
        dict with 'input_effective_rank', 'hidden_effective_rank',
        'compression_ratio', 'd_model', 'd_mlp'.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_mid", layer)]  # [seq, d_model]
    hidden = cache[("post", layer)]  # [seq, d_mlp]

    def _effective_rank(X):
        s = jnp.linalg.svd(X, compute_uv=False)
        s = s / (jnp.sum(s) + 1e-10)
        s = jnp.where(s > 1e-10, s, 1e-10)
        return float(jnp.exp(-jnp.sum(s * jnp.log(s))))

    input_rank = _effective_rank(resid)
    hidden_rank = _effective_rank(hidden)
    d_model = resid.shape[-1]
    d_mlp = hidden.shape[-1]
    return {
        "input_effective_rank": input_rank,
        "hidden_effective_rank": hidden_rank,
        "compression_ratio": hidden_rank / (input_rank + 1e-10),
        "d_model": int(d_model),
        "d_mlp": int(d_mlp),
    }


def mlp_hidden_utilization(model, tokens, layer=0):
    """What fraction of MLP hidden neurons are actively used?

    Returns:
        dict with 'active_fraction', 'n_active', 'n_total',
        'mean_activation', 'per_neuron_activity' array.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    activity = jnp.mean(jnp.abs(hidden), axis=0)  # [d_mlp]
    threshold = float(jnp.mean(activity)) * 0.01
    active = activity > threshold
    n_active = int(jnp.sum(active))
    n_total = int(hidden.shape[-1])
    return {
        "active_fraction": n_active / n_total,
        "n_active": n_active,
        "n_total": n_total,
        "mean_activation": float(jnp.mean(activity)),
        "per_neuron_activity": activity,
    }


def mlp_input_reconstruction(model, tokens, layer=0):
    """How well can the MLP output reconstruct the input direction?

    Measures cosine similarity between MLP input and output.

    Returns:
        dict with 'per_position' list, 'mean_reconstruction'.
    """
    _, cache = model.run_with_cache(tokens)
    resid_mid = cache[("resid_mid", layer)]  # [seq, d_model]
    mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
    seq_len = resid_mid.shape[0]
    per_position = []
    for pos in range(seq_len):
        inp = resid_mid[pos]
        out = mlp_out[pos]
        cos = float(jnp.dot(inp, out) / (jnp.linalg.norm(inp) * jnp.linalg.norm(out) + 1e-10))
        per_position.append({
            "position": pos,
            "cosine": cos,
        })
    mean_recon = sum(p["cosine"] for p in per_position) / len(per_position)
    return {
        "per_position": per_position,
        "mean_reconstruction": mean_recon,
    }


def mlp_expansion_selectivity(model, tokens, layer=0, top_k=5):
    """Which hidden neurons are most selective (activate for few positions)?

    Returns:
        dict with 'most_selective' top-k neurons, 'least_selective',
        'mean_selectivity'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    max_per_neuron = jnp.max(jnp.abs(hidden), axis=0)  # [d_mlp]
    safe_max = jnp.where(max_per_neuron > 1e-10, max_per_neuron, 1.0)
    normed = jnp.abs(hidden) / safe_max[None, :]  # [seq, d_mlp]
    selectivity = 1.0 - jnp.mean(normed, axis=0)  # high = few positions active
    k = min(top_k, hidden.shape[-1])
    most_idx = jnp.argsort(-selectivity)[:k]
    least_idx = jnp.argsort(selectivity)[:k]
    most_selective = [(int(i), float(selectivity[i])) for i in most_idx]
    least_selective = [(int(i), float(selectivity[i])) for i in least_idx]
    return {
        "most_selective": most_selective,
        "least_selective": least_selective,
        "mean_selectivity": float(jnp.mean(selectivity)),
    }


def mlp_bottleneck_summary(model, tokens):
    """Summary of MLP bottleneck analysis across all layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        comp = mlp_compression_ratio(model, tokens, layer=layer)
        util = mlp_hidden_utilization(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "compression_ratio": comp["compression_ratio"],
            "active_fraction": util["active_fraction"],
            "mean_activation": util["mean_activation"],
        })
    return {"per_layer": per_layer}
