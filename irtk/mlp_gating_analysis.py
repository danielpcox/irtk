"""MLP gating analysis: analyze how MLP gating controls information flow.

Study the pre-activation gate patterns, activation sparsity,
and how gating selects which information passes through the MLP.
"""

import jax.numpy as jnp


def mlp_activation_sparsity(model, tokens, layer=0):
    """Measure sparsity of MLP post-activation values.

    Returns:
        dict with 'sparsity' (fraction near zero), 'mean_magnitude',
        'per_position' list.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    seq_len = hidden.shape[0]
    threshold = float(jnp.mean(jnp.abs(hidden))) * 0.01
    per_position = []
    for pos in range(seq_len):
        h = hidden[pos]
        n_zero = int(jnp.sum(jnp.abs(h) < threshold))
        n_total = int(h.shape[0])
        per_position.append({
            "position": pos,
            "sparsity": n_zero / n_total,
            "mean_magnitude": float(jnp.mean(jnp.abs(h))),
        })
    overall_sparsity = sum(p["sparsity"] for p in per_position) / len(per_position)
    return {
        "sparsity": overall_sparsity,
        "mean_magnitude": float(jnp.mean(jnp.abs(hidden))),
        "per_position": per_position,
    }


def mlp_pre_post_correlation(model, tokens, layer=0):
    """Correlation between pre-activation and post-activation.

    Shows how much the activation function changes the representation.

    Returns:
        dict with 'mean_correlation', 'per_position' list.
    """
    _, cache = model.run_with_cache(tokens)
    pre = cache[("pre", layer)]  # [seq, d_mlp]
    post = cache[("post", layer)]  # [seq, d_mlp]
    seq_len = pre.shape[0]
    per_position = []
    for pos in range(seq_len):
        p = pre[pos]
        q = post[pos]
        cos = float(jnp.dot(p, q) / (jnp.linalg.norm(p) * jnp.linalg.norm(q) + 1e-10))
        per_position.append({
            "position": pos,
            "correlation": cos,
        })
    mean_corr = sum(p["correlation"] for p in per_position) / len(per_position)
    return {
        "mean_correlation": mean_corr,
        "per_position": per_position,
    }


def mlp_activation_distribution(model, tokens, layer=0):
    """Distribution statistics of MLP activations.

    Returns:
        dict with mean, std, skewness, kurtosis of post-activations.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]
    flat = post.reshape(-1)
    mean = float(jnp.mean(flat))
    std = float(jnp.std(flat))
    centered = flat - mean
    skew = float(jnp.mean(centered ** 3) / (std ** 3 + 1e-10))
    kurt = float(jnp.mean(centered ** 4) / (std ** 4 + 1e-10) - 3.0)
    return {
        "mean": mean,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "min": float(jnp.min(flat)),
        "max": float(jnp.max(flat)),
    }


def mlp_gating_selectivity(model, tokens, layer=0, top_k=5):
    """Which neurons are most selectively gated (on for few positions)?

    Returns:
        dict with 'most_selective', 'least_selective', 'mean_selectivity'.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]
    active = (jnp.abs(post) > 1e-6).astype(jnp.float32)  # [seq, d_mlp]
    activation_rate = jnp.mean(active, axis=0)  # [d_mlp]
    selectivity = 1.0 - activation_rate
    k = min(top_k, post.shape[-1])
    most_idx = jnp.argsort(-selectivity)[:k]
    least_idx = jnp.argsort(selectivity)[:k]
    most_selective = [(int(i), float(selectivity[i])) for i in most_idx]
    least_selective = [(int(i), float(selectivity[i])) for i in least_idx]
    return {
        "most_selective": most_selective,
        "least_selective": least_selective,
        "mean_selectivity": float(jnp.mean(selectivity)),
    }


def mlp_gating_summary(model, tokens):
    """Summary of MLP gating analysis across all layers.

    Returns:
        dict with 'per_layer' list.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        sparsity = mlp_activation_sparsity(model, tokens, layer=layer)
        dist = mlp_activation_distribution(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "sparsity": sparsity["sparsity"],
            "mean_magnitude": sparsity["mean_magnitude"],
            "skewness": dist["skewness"],
        })
    return {"per_layer": per_layer}
