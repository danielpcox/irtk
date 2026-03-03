"""MLP activation profiling: characterize the hidden layer activations of MLPs."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_pre_activation_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile the pre-activation (before nonlinearity) values across layers.

    Shows magnitude and distribution of MLP inputs.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        pre = cache[f'blocks.{layer}.mlp.hook_pre']  # [seq, d_mlp]
        norms = jnp.linalg.norm(pre, axis=-1)  # [seq]

        # Per-neuron statistics
        mean_act = jnp.mean(pre, axis=0)  # [d_mlp]
        std_act = jnp.std(pre, axis=0)

        per_layer.append({
            'layer': layer,
            'mean_norm': float(jnp.mean(norms)),
            'std_norm': float(jnp.std(norms)),
            'mean_activation': float(jnp.mean(pre)),
            'std_activation': float(jnp.std(pre)),
            'max_activation': float(jnp.max(jnp.abs(pre))),
            'fraction_positive': float(jnp.mean(pre > 0)),
        })

    return {
        'per_layer': per_layer,
    }


def mlp_post_activation_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile the post-activation (after nonlinearity) values across layers.

    Shows sparsity and magnitude after the activation function.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
        norms = jnp.linalg.norm(post, axis=-1)

        # Sparsity
        near_zero = float(jnp.mean(jnp.abs(post) < 1e-5))

        per_layer.append({
            'layer': layer,
            'mean_norm': float(jnp.mean(norms)),
            'std_norm': float(jnp.std(norms)),
            'mean_activation': float(jnp.mean(post)),
            'max_activation': float(jnp.max(post)),
            'sparsity': near_zero,
            'fraction_positive': float(jnp.mean(post > 0)),
        })

    return {
        'per_layer': per_layer,
    }


def mlp_neuron_activation_distribution(model: HookedTransformer, tokens: jnp.ndarray, layer: int, top_k: int = 10) -> dict:
    """Distribution of individual neuron activations at a layer.

    Identifies the most active and most inactive neurons.
    """
    _, cache = model.run_with_cache(tokens)

    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    d_mlp = post.shape[-1]

    # Mean activation per neuron across positions
    mean_per_neuron = jnp.mean(post, axis=0)  # [d_mlp]
    max_per_neuron = jnp.max(jnp.abs(post), axis=0)

    # Top active neurons
    top_indices = jnp.argsort(mean_per_neuron)[-top_k:][::-1]
    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append({
            'neuron': idx,
            'mean_activation': float(mean_per_neuron[idx]),
            'max_activation': float(max_per_neuron[idx]),
        })

    # Dead neurons
    dead_mask = jnp.all(jnp.abs(post) < 1e-5, axis=0)
    n_dead = int(jnp.sum(dead_mask))

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'top_neurons': top_neurons,
        'n_dead': n_dead,
        'dead_fraction': n_dead / d_mlp,
        'mean_activation': float(jnp.mean(post)),
    }


def mlp_activation_position_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How do MLP activations vary across positions?

    Shows per-position activation magnitude and sparsity.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]

    per_position = []
    for pos in range(seq_len):
        act = post[pos]  # [d_mlp]
        norm = float(jnp.linalg.norm(act))
        sparsity = float(jnp.mean(jnp.abs(act) < 1e-5))
        n_active = int(jnp.sum(jnp.abs(act) > 1e-5))

        per_position.append({
            'position': pos,
            'token': int(tokens[pos]),
            'activation_norm': norm,
            'sparsity': sparsity,
            'n_active_neurons': n_active,
            'max_activation': float(jnp.max(jnp.abs(act))),
        })

    return {
        'layer': layer,
        'per_position': per_position,
    }


def mlp_pre_post_correlation(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Correlation between pre-activation magnitude and post-activation magnitude.

    Shows how the nonlinearity transforms the signal.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        pre = cache[f'blocks.{layer}.mlp.hook_pre']  # [seq, d_mlp]
        post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]

        pre_norms = jnp.linalg.norm(pre, axis=-1)  # [seq]
        post_norms = jnp.linalg.norm(post, axis=-1)

        # Correlation
        pre_centered = pre_norms - jnp.mean(pre_norms)
        post_centered = post_norms - jnp.mean(post_norms)
        corr = float(jnp.dot(pre_centered, post_centered) / (
            jnp.linalg.norm(pre_centered) * jnp.linalg.norm(post_centered) + 1e-10
        ))

        # Compression ratio
        compression = float(jnp.mean(post_norms) / (jnp.mean(pre_norms) + 1e-10))

        per_layer.append({
            'layer': layer,
            'norm_correlation': corr,
            'compression_ratio': compression,
            'pre_mean_norm': float(jnp.mean(pre_norms)),
            'post_mean_norm': float(jnp.mean(post_norms)),
        })

    return {
        'per_layer': per_layer,
    }
