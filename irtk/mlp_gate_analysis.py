"""MLP gate analysis: analyze MLP gating and activation patterns."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_activation_distribution(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Analyze the distribution of MLP activations.

    Measures activation statistics: sparsity, magnitude distribution, etc.
    """
    _, cache = model.run_with_cache(tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    post_acts = cache[post_key]  # [seq, d_mlp]
    seq_len, d_mlp = post_acts.shape

    # Overall statistics
    mean_act = float(jnp.mean(post_acts))
    std_act = float(jnp.std(post_acts))
    max_act = float(jnp.max(post_acts))
    min_act = float(jnp.min(post_acts))

    # Sparsity (fraction of zero/negative activations for ReLU-like)
    active_frac = float(jnp.mean(post_acts > 0))

    # Per-neuron mean activation
    neuron_means = jnp.mean(post_acts, axis=0)
    n_dead = int(jnp.sum(jnp.max(post_acts, axis=0) <= 0))

    return {
        'layer': layer,
        'mean_activation': mean_act,
        'std_activation': std_act,
        'max_activation': max_act,
        'min_activation': min_act,
        'active_fraction': active_frac,
        'n_dead_neurons': n_dead,
        'dead_fraction': n_dead / d_mlp,
    }


def mlp_pre_post_relationship(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Analyze the relationship between pre and post activation values.

    Shows how the activation function transforms the input.
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f'blocks.{layer}.mlp.hook_pre'
    post_key = f'blocks.{layer}.mlp.hook_post'

    pre = cache[pre_key]   # [seq, d_mlp]
    post = cache[post_key]  # [seq, d_mlp]

    seq_len, d_mlp = pre.shape

    # Correlation between pre and post
    pre_flat = pre.reshape(-1)
    post_flat = post.reshape(-1)
    correlation = float(jnp.corrcoef(pre_flat, post_flat)[0, 1])

    # Ratio statistics
    nonzero = jnp.abs(pre_flat) > 1e-8
    ratios = jnp.where(nonzero, post_flat / (pre_flat + 1e-10), 0.0)
    mean_ratio = float(jnp.mean(jnp.abs(ratios[nonzero]))) if jnp.any(nonzero) else 0.0

    # Fraction passed (post > 0 when pre > 0)
    positive_pre = pre_flat > 0
    if jnp.any(positive_pre):
        pass_rate = float(jnp.mean(post_flat[positive_pre] > 0))
    else:
        pass_rate = 0.0

    # Suppression (pre > 0 but post ≈ 0)
    suppressed = jnp.sum((pre_flat > 0.01) & (post_flat < 0.001))
    suppression_rate = float(suppressed / jnp.sum(pre_flat > 0.01)) if jnp.sum(pre_flat > 0.01) > 0 else 0.0

    return {
        'layer': layer,
        'correlation': correlation,
        'mean_ratio': mean_ratio,
        'pass_rate': pass_rate,
        'suppression_rate': suppression_rate,
    }


def neuron_activation_frequency(model: HookedTransformer, tokens: jnp.ndarray, layer: int, top_k: int = 10) -> dict:
    """How frequently does each neuron fire across positions?

    Finds the most and least active neurons.
    """
    _, cache = model.run_with_cache(tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    post = cache[post_key]  # [seq, d_mlp]
    seq_len, d_mlp = post.shape

    # Per-neuron activation frequency
    active_counts = jnp.sum(post > 0, axis=0)  # [d_mlp]
    frequencies = active_counts / seq_len

    # Top-k most active
    most_active_idx = jnp.argsort(frequencies)[-top_k:][::-1]
    most_active = []
    for idx in most_active_idx:
        i = int(idx)
        most_active.append({
            'neuron_idx': i,
            'frequency': float(frequencies[i]),
            'mean_activation': float(jnp.mean(post[:, i])),
        })

    # Top-k least active (non-dead)
    alive = frequencies > 0
    if jnp.any(alive):
        alive_freq = jnp.where(alive, frequencies, jnp.inf)
        least_active_idx = jnp.argsort(alive_freq)[:top_k]
        least_active = []
        for idx in least_active_idx:
            i = int(idx)
            if float(frequencies[i]) > 0:
                least_active.append({
                    'neuron_idx': i,
                    'frequency': float(frequencies[i]),
                    'mean_activation': float(jnp.mean(post[:, i])),
                })
    else:
        least_active = []

    return {
        'layer': layer,
        'most_active': most_active,
        'least_active': least_active,
        'mean_frequency': float(jnp.mean(frequencies)),
        'n_always_active': int(jnp.sum(frequencies >= 1.0)),
        'n_never_active': int(jnp.sum(frequencies == 0)),
    }


def mlp_output_direction_analysis(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Analyze the output directions of the MLP.

    Measures how the MLP output direction varies across positions.
    """
    _, cache = model.run_with_cache(tokens)

    mlp_out_key = f'blocks.{layer}.hook_mlp_out'
    mlp_out = cache[mlp_out_key]  # [seq, d_model]
    seq_len = mlp_out.shape[0]

    # Norms
    norms = jnp.linalg.norm(mlp_out, axis=-1)
    mean_norm = float(jnp.mean(norms))

    # Direction consistency
    normed = mlp_out / (norms[:, None] + 1e-10)
    mean_dir = jnp.mean(normed, axis=0)
    mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)

    per_position = []
    for pos in range(seq_len):
        alignment = float(jnp.dot(normed[pos], mean_dir))
        per_position.append({
            'position': pos,
            'norm': float(norms[pos]),
            'alignment_to_mean': alignment,
        })

    mean_alignment = float(jnp.mean(jnp.array([p['alignment_to_mean'] for p in per_position])))

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_norm': mean_norm,
        'mean_alignment': mean_alignment,
        'is_consistent': mean_alignment > 0.5,
    }


def mlp_contribution_vs_attention(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Compare MLP and attention contributions at each layer.

    Shows the balance of information processing between the two.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']

        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=-1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))

        # Cosine alignment between attn and MLP outputs
        attn_normed = attn_out / (jnp.linalg.norm(attn_out, axis=-1, keepdims=True) + 1e-10)
        mlp_normed = mlp_out / (jnp.linalg.norm(mlp_out, axis=-1, keepdims=True) + 1e-10)
        alignment = float(jnp.mean(jnp.sum(attn_normed * mlp_normed, axis=-1)))

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'mlp_fraction': mlp_norm / (attn_norm + mlp_norm + 1e-10),
            'alignment': alignment,
            'cooperate': alignment > 0,
        })

    n_mlp_dominant = sum(1 for p in per_layer if p['mlp_fraction'] > 0.5)

    return {
        'per_layer': per_layer,
        'n_mlp_dominant': n_mlp_dominant,
        'n_attn_dominant': n_layers - n_mlp_dominant,
    }
