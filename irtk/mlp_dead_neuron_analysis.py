"""MLP dead neuron analysis: identifying and characterizing inactive neurons."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def dead_neuron_detection(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0, threshold: float = 1e-6) -> dict:
    """Identify neurons that never activate above threshold.

    Dead neurons waste capacity without contributing to computation.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]

    max_activations = jnp.max(post, axis=0)  # [d_mlp]
    is_dead = max_activations < threshold
    n_dead = int(jnp.sum(is_dead))
    total = post.shape[1]

    dead_indices = [int(i) for i in jnp.where(is_dead)[0][:20]]

    return {
        "layer": layer,
        "n_dead": n_dead,
        "total_neurons": total,
        "dead_fraction": n_dead / total,
        "dead_indices": dead_indices,
        "threshold": threshold,
    }


def neuron_activation_frequency(model: HookedTransformer, tokens: jnp.ndarray,
                                   layer: int = 0) -> dict:
    """How often does each neuron fire across positions?"""
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]
    seq_len = post.shape[0]

    fire_count = jnp.sum((post > 0).astype(jnp.float32), axis=0)  # [d_mlp]
    frequency = fire_count / seq_len

    # Histogram of frequencies
    bins = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
    histogram = {}
    for i in range(len(bins) - 1):
        count = int(jnp.sum((frequency >= bins[i]) & (frequency < bins[i + 1])))
        histogram[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = count
    # Handle exactly 1.0
    histogram["1.0"] = int(jnp.sum(frequency >= 1.0))

    return {
        "layer": layer,
        "mean_frequency": float(jnp.mean(frequency)),
        "median_frequency": float(jnp.median(frequency)),
        "histogram": histogram,
        "n_always_active": int(jnp.sum(frequency >= 1.0)),
        "n_never_active": int(jnp.sum(frequency == 0)),
    }


def neuron_activation_magnitude_distribution(model: HookedTransformer, tokens: jnp.ndarray,
                                                layer: int = 0) -> dict:
    """Distribution of activation magnitudes across neurons."""
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]

    mean_acts = jnp.mean(post, axis=0)  # [d_mlp]
    max_acts = jnp.max(post, axis=0)
    std_acts = jnp.std(post, axis=0)

    return {
        "layer": layer,
        "global_mean": float(jnp.mean(mean_acts)),
        "global_max": float(jnp.max(max_acts)),
        "mean_std": float(jnp.mean(std_acts)),
        "n_high_variance": int(jnp.sum(std_acts > float(jnp.mean(std_acts)) * 2)),
    }


def near_dead_neurons(model: HookedTransformer, tokens: jnp.ndarray,
                         layer: int = 0, percentile: float = 10.0) -> dict:
    """Neurons with very low activation that may be nearly dead."""
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]

    mean_acts = jnp.mean(jnp.abs(post), axis=0)  # [d_mlp]
    threshold = float(jnp.percentile(mean_acts, percentile))

    near_dead = mean_acts < threshold
    n_near_dead = int(jnp.sum(near_dead))

    # Get weakest neurons
    weakest = jnp.argsort(mean_acts)[:10]
    weakest_info = []
    for idx in weakest:
        idx_int = int(idx)
        weakest_info.append({
            "neuron": idx_int,
            "mean_activation": float(mean_acts[idx_int]),
        })

    return {
        "layer": layer,
        "n_near_dead": n_near_dead,
        "threshold": threshold,
        "weakest_neurons": weakest_info,
    }


def dead_neuron_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer dead neuron summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        dead = dead_neuron_detection(model, tokens, layer)
        freq = neuron_activation_frequency(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "dead_fraction": dead["dead_fraction"],
            "mean_frequency": freq["mean_frequency"],
            "n_never_active": freq["n_never_active"],
        })
    return {"per_layer": per_layer}
