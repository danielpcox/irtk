"""MLP superposition detection: identify superposed representations in MLPs.

Detect when MLP neurons encode multiple features (superposition),
measure interference between features, and analyze capacity utilization.
"""

import jax.numpy as jnp


def neuron_activation_correlation(model, tokens, layer=0):
    """Correlation matrix between neuron activations.

    High off-diagonal correlations suggest shared features or superposition.

    Returns:
        dict with 'correlation_matrix', 'mean_off_diagonal', 'max_correlation'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
    norms = jnp.linalg.norm(centered, axis=0, keepdims=True) + 1e-10
    normed = centered / norms
    corr = normed.T @ normed / hidden.shape[0]  # [d_mlp, d_mlp]
    d_mlp = corr.shape[0]
    mask = jnp.ones((d_mlp, d_mlp)) - jnp.eye(d_mlp)
    off_diag = jnp.abs(corr) * mask
    mean_off = float(jnp.sum(off_diag) / jnp.sum(mask))
    max_corr = float(jnp.max(off_diag))
    return {
        "correlation_matrix": corr,
        "mean_off_diagonal": mean_off,
        "max_correlation": max_corr,
    }


def neuron_output_interference(model, tokens, layer=0, top_k=5):
    """Find neuron pairs whose output directions interfere.

    High cosine similarity between W_out columns suggests superposition.

    Returns:
        dict with 'most_interfering' top-k pairs, 'mean_interference'.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    norms = jnp.linalg.norm(W_out, axis=-1, keepdims=True) + 1e-10
    normed = W_out / norms
    sim = normed @ normed.T  # [d_mlp, d_mlp]
    d_mlp = sim.shape[0]
    mask = jnp.ones((d_mlp, d_mlp)) - jnp.eye(d_mlp)
    masked_sim = jnp.abs(sim) * mask
    flat = masked_sim.reshape(-1)
    top_indices = jnp.argsort(-flat)[:top_k]
    pairs = []
    for idx in top_indices:
        i = int(idx // d_mlp)
        j = int(idx % d_mlp)
        pairs.append({
            "neuron_a": i,
            "neuron_b": j,
            "interference": float(masked_sim[i, j]),
        })
    mean_interf = float(jnp.sum(masked_sim) / jnp.sum(mask))
    return {
        "most_interfering": pairs,
        "mean_interference": mean_interf,
    }


def neuron_polysemanticity(model, tokens, layer=0, top_k=5):
    """Measure how polysemantic each neuron is.

    A polysemantic neuron activates for diverse input patterns.
    Uses coefficient of variation of activations across positions.

    Returns:
        dict with 'per_neuron_scores' array, 'most_polysemantic' top-k,
        'mean_polysemanticity'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    means = jnp.mean(jnp.abs(hidden), axis=0)
    stds = jnp.std(hidden, axis=0)
    cv = stds / (means + 1e-10)
    poly_score = 1.0 / (1.0 + cv)
    k = min(top_k, hidden.shape[-1])
    top_idx = jnp.argsort(-poly_score)[:k]
    most_poly = [(int(i), float(poly_score[i])) for i in top_idx]
    return {
        "per_neuron_scores": poly_score,
        "most_polysemantic": most_poly,
        "mean_polysemanticity": float(jnp.mean(poly_score)),
    }


def superposition_dimensionality(model, tokens, layer=0):
    """Estimate the effective dimensionality of MLP representations.

    Compares the effective rank to the actual dimension to estimate
    how much superposition is being used.

    Returns:
        dict with 'effective_rank', 'd_mlp', 'superposition_ratio'.
    """
    _, cache = model.run_with_cache(tokens)
    hidden = cache[("post", layer)]  # [seq, d_mlp]
    s = jnp.linalg.svd(hidden, compute_uv=False)
    s = s / (jnp.sum(s) + 1e-10)
    s = jnp.where(s > 1e-10, s, 1e-10)
    eff_rank = float(jnp.exp(-jnp.sum(s * jnp.log(s))))
    d_mlp = int(hidden.shape[-1])
    return {
        "effective_rank": eff_rank,
        "d_mlp": d_mlp,
        "superposition_ratio": d_mlp / (eff_rank + 1e-10),
    }


def mlp_superposition_summary(model, tokens):
    """Summary of superposition detection across all layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        corr = neuron_activation_correlation(model, tokens, layer=layer)
        dim = superposition_dimensionality(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "mean_correlation": corr["mean_off_diagonal"],
            "superposition_ratio": dim["superposition_ratio"],
            "effective_rank": dim["effective_rank"],
        })
    return {"per_layer": per_layer}
