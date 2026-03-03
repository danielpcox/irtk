"""MLP neuron profiling.

Detailed per-neuron behavior profiling: activation patterns,
selectivity, logit impact, and feature clustering.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def neuron_activation_profile(model, tokens, layer=0, top_k=5):
    """Profile each neuron's activation pattern across positions.

    Returns:
        dict with per_neuron activation statistics.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'top_neurons': [], 'n_active': 0}

    activations = cache[post_key]  # [seq, d_mlp]
    d_mlp = activations.shape[1]

    # Per-neuron stats
    mean_act = jnp.mean(activations, axis=0)  # [d_mlp]
    max_act = jnp.max(activations, axis=0)
    sparsity = jnp.mean(activations > 0.01, axis=0)  # fraction of positions active

    # Top neurons by max activation
    top_indices = jnp.argsort(-max_act)[:top_k]

    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append({
            'neuron': idx,
            'mean_activation': float(mean_act[idx]),
            'max_activation': float(max_act[idx]),
            'sparsity': float(sparsity[idx]),
            'activations': [float(activations[p, idx]) for p in range(seq_len)],
        })

    n_active = int(jnp.sum(max_act > 0.01))

    return {
        'layer': layer,
        'top_neurons': top_neurons,
        'n_active': n_active,
        'n_total': d_mlp,
        'mean_sparsity': float(jnp.mean(sparsity)),
    }


def neuron_logit_impact(model, tokens, layer=0, pos=-1, top_k=5):
    """Which neurons most affect the target logit?

    Returns:
        dict with per_neuron logit contribution.
    """
    cache = _run_and_cache(model, tokens)
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'top_neurons': []}

    activations = cache[post_key]  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    logits = model(tokens)
    target = int(jnp.argmax(logits[pos]))
    W_U_col = W_U[:, target]  # [d_model]

    # Per-neuron logit contribution = activation[pos, n] * (W_out[n] @ W_U_col)
    direction_logit = W_out @ W_U_col  # [d_mlp]
    contributions = activations[pos] * direction_logit  # [d_mlp]

    top_indices = jnp.argsort(-jnp.abs(contributions))[:top_k]

    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append({
            'neuron': idx,
            'activation': float(activations[pos, idx]),
            'logit_contribution': float(contributions[idx]),
            'weight_alignment': float(direction_logit[idx]),
        })

    return {
        'layer': layer,
        'target_token': target,
        'top_neurons': top_neurons,
        'total_logit': float(jnp.sum(contributions)),
    }


def neuron_selectivity(model, tokens, layer=0, top_k=5):
    """How selective is each neuron (fires for few positions)?

    Returns:
        dict with selectivity metrics.
    """
    cache = _run_and_cache(model, tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'most_selective': [], 'least_selective': []}

    activations = cache[post_key]  # [seq, d_mlp]
    d_mlp = activations.shape[1]
    seq_len = activations.shape[0]

    # Selectivity = 1 - (fraction of positions with nonzero activation)
    active_fraction = jnp.mean(activations > 0.01, axis=0)  # [d_mlp]
    selectivity = 1.0 - active_fraction

    # Most selective (fires rarely)
    most_sel_idx = jnp.argsort(-selectivity)[:top_k]
    most_selective = []
    for idx in most_sel_idx:
        idx = int(idx)
        most_selective.append({
            'neuron': idx,
            'selectivity': float(selectivity[idx]),
            'n_active_positions': int(jnp.sum(activations[:, idx] > 0.01)),
        })

    # Least selective (fires everywhere)
    least_sel_idx = jnp.argsort(selectivity)[:top_k]
    least_selective = []
    for idx in least_sel_idx:
        idx = int(idx)
        least_selective.append({
            'neuron': idx,
            'selectivity': float(selectivity[idx]),
            'n_active_positions': int(jnp.sum(activations[:, idx] > 0.01)),
        })

    return {
        'layer': layer,
        'most_selective': most_selective,
        'least_selective': least_selective,
        'mean_selectivity': float(jnp.mean(selectivity)),
    }


def neuron_correlation_clusters(model, tokens, layer=0, threshold=0.7):
    """Find groups of neurons that activate together.

    Returns:
        dict with correlated neuron pairs.
    """
    cache = _run_and_cache(model, tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'correlated_pairs': []}

    activations = cache[post_key]  # [seq, d_mlp]
    d_mlp = activations.shape[1]

    # Only consider neurons that actually fire
    active_mask = jnp.max(activations, axis=0) > 0.01  # [d_mlp]
    active_indices = jnp.where(active_mask)[0]

    if len(active_indices) < 2:
        return {'correlated_pairs': [], 'n_active': len(active_indices)}

    # Subsample if too many
    max_check = min(len(active_indices), 50)
    check_indices = active_indices[:max_check]

    pairs = []
    for i in range(len(check_indices)):
        for j in range(i + 1, len(check_indices)):
            ni = int(check_indices[i])
            nj = int(check_indices[j])
            a_i = activations[:, ni]
            a_j = activations[:, nj]
            cos = float(jnp.dot(a_i, a_j) /
                       jnp.maximum(jnp.linalg.norm(a_i) * jnp.linalg.norm(a_j), 1e-10))
            if abs(cos) > threshold:
                pairs.append({
                    'neuron_i': ni,
                    'neuron_j': nj,
                    'correlation': cos,
                })

    return {
        'layer': layer,
        'correlated_pairs': pairs,
        'n_active': len(active_indices),
    }


def dead_neuron_analysis(model, tokens, layer=0, threshold=0.001):
    """Identify dead or near-dead neurons.

    Returns:
        dict with dead neuron info.
    """
    cache = _run_and_cache(model, tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'n_dead': 0, 'dead_fraction': 0.0}

    activations = cache[post_key]  # [seq, d_mlp]
    d_mlp = activations.shape[1]

    max_act = jnp.max(jnp.abs(activations), axis=0)  # [d_mlp]
    dead_mask = max_act < threshold
    n_dead = int(jnp.sum(dead_mask))

    # Near-dead: barely fire
    near_dead_mask = (max_act >= threshold) & (max_act < threshold * 10)
    n_near_dead = int(jnp.sum(near_dead_mask))

    return {
        'layer': layer,
        'n_dead': n_dead,
        'n_near_dead': n_near_dead,
        'n_healthy': d_mlp - n_dead - n_near_dead,
        'dead_fraction': n_dead / d_mlp,
        'total_neurons': d_mlp,
    }
