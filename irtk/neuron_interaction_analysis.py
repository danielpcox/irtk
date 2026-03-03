"""Neuron interaction analysis.

Analyze how MLP neurons interact: pairwise interference, cooperative groups,
compensation patterns, context dependencies, and ensemble effects.
"""

import jax
import jax.numpy as jnp


def neuron_coactivation_matrix(model, tokens, layer):
    """Compute pairwise neuron coactivation patterns.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer to analyze

    Returns:
        dict with coactivation matrix and statistics.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.hook_mlp_out']
    # Get pre-activation to see neuron activations
    pre_key = f'blocks.{layer}.mlp.hook_pre'
    post_key = f'blocks.{layer}.mlp.hook_post'
    if pre_key in cache:
        mlp_pre = cache[pre_key]
    elif post_key in cache:
        mlp_pre = cache[post_key]
    else:
        return {'error': 'no neuron activations found'}

    # mlp_pre: [seq, d_mlp]
    # Binarize: active = > 0
    active = (mlp_pre > 0).astype(jnp.float32)  # [seq, d_mlp]
    n_neurons = active.shape[-1]

    # Coactivation: fraction of positions where both active
    # Use smaller subset if too many neurons
    max_neurons = min(n_neurons, 64)
    active_sub = active[:, :max_neurons]
    coact = (active_sub.T @ active_sub) / active.shape[0]  # [max_neurons, max_neurons]

    # Find most coactivated pair (off-diagonal)
    mask = 1.0 - jnp.eye(max_neurons)
    coact_masked = coact * mask
    flat_idx = int(jnp.argmax(coact_masked))
    i, j = flat_idx // max_neurons, flat_idx % max_neurons

    return {
        'coactivation_matrix': coact,
        'n_neurons_analyzed': max_neurons,
        'mean_coactivation': float(jnp.mean(coact_masked)),
        'max_coactivated_pair': (int(i), int(j)),
        'max_coactivation': float(coact[i, j]),
    }


def neuron_interference(model, tokens, layer, top_k=5):
    """Measure how ablating one neuron affects others' contributions.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        top_k: number of neurons to analyze

    Returns:
        dict with interference scores.
    """
    _, cache = model.run_with_cache(tokens)
    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'error': 'no post-activation found'}
    mlp_post = cache[post_key]

    n_neurons = mlp_post.shape[-1]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    # Get per-neuron contributions
    contributions = []
    for n in range(n_neurons):
        c = float(jnp.mean(jnp.abs(mlp_post[:, n])))
        contributions.append(c)
    contributions = jnp.array(contributions)

    # Find top_k most active neurons
    top_indices = jnp.argsort(-contributions)[:top_k]

    interference_scores = []
    for idx in top_indices:
        idx = int(idx)
        # Neuron's output direction
        direction = W_out[idx]
        direction_norm = direction / (jnp.linalg.norm(direction) + 1e-10)

        # How much do other neurons' W_out align with this direction?
        alignments = W_out @ direction_norm  # [d_mlp]
        # Interference = sum of absolute alignment * activation
        mean_acts = jnp.mean(jnp.abs(mlp_post), axis=0)  # [d_mlp]
        interference = float(jnp.sum(jnp.abs(alignments) * mean_acts) - jnp.abs(alignments[idx]) * mean_acts[idx])

        interference_scores.append({
            'neuron_idx': idx,
            'activation': float(contributions[idx]),
            'total_interference': interference,
            'n_interfering': int(jnp.sum(jnp.abs(alignments) > 0.1) - 1),
        })

    interference_scores.sort(key=lambda s: -s['total_interference'])
    return {
        'per_neuron': interference_scores,
        'most_interfered': interference_scores[0] if interference_scores else None,
    }


def cooperative_neuron_groups(model, tokens, layer, threshold=0.5):
    """Find groups of neurons that consistently activate together.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        threshold: coactivation threshold for grouping

    Returns:
        dict with cooperative groups.
    """
    _, cache = model.run_with_cache(tokens)
    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'error': 'no post-activation found'}
    mlp_post = cache[post_key]

    n_neurons = min(mlp_post.shape[-1], 64)
    acts = mlp_post[:, :n_neurons]

    # Correlation matrix
    mean_acts = jnp.mean(acts, axis=0, keepdims=True)
    centered = acts - mean_acts
    norms = jnp.linalg.norm(centered, axis=0, keepdims=True) + 1e-10
    normalized = centered / norms
    corr = (normalized.T @ normalized) / acts.shape[0]

    # Find groups via thresholding
    groups = []
    visited = set()
    for i in range(n_neurons):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n_neurons):
            if j not in visited and float(corr[i, j]) > threshold:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    return {
        'groups': groups,
        'n_groups': len(groups),
        'mean_correlation': float(jnp.mean(jnp.abs(corr) - jnp.eye(n_neurons))),
        'n_neurons_analyzed': n_neurons,
    }


def neuron_compensation(model, tokens, layer, neuron_idx):
    """Measure how well other neurons compensate when one is ablated.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        neuron_idx: neuron to ablate

    Returns:
        dict with compensation analysis.
    """
    # Clean run
    clean_logits = model(tokens)

    # Ablate neuron
    hook_name = f'blocks.{layer}.mlp.hook_post'

    def zero_neuron(x, name):
        return x.at[:, neuron_idx].set(0.0)

    mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_neuron)])

    # Measure effect
    logit_diff = mod_logits - clean_logits
    max_effect = float(jnp.max(jnp.abs(logit_diff)))
    mean_effect = float(jnp.mean(jnp.abs(logit_diff)))

    # Check if predictions change
    clean_preds = jnp.argmax(clean_logits, axis=-1)
    mod_preds = jnp.argmax(mod_logits, axis=-1)
    pred_changed = int(jnp.sum(clean_preds != mod_preds))

    # Get neuron's contribution norm
    _, cache = model.run_with_cache(tokens)
    post_key = f'blocks.{layer}.mlp.hook_post'
    neuron_activation = float(jnp.mean(jnp.abs(cache[post_key][:, neuron_idx]))) if post_key in cache else 0.0

    # Compensation ratio: small effect despite large activation = high compensation
    compensation_ratio = 1.0 - (mean_effect / max(neuron_activation, 1e-10))

    return {
        'neuron_idx': neuron_idx,
        'neuron_activation': neuron_activation,
        'max_logit_effect': max_effect,
        'mean_logit_effect': mean_effect,
        'predictions_changed': pred_changed,
        'compensation_ratio': max(0.0, min(1.0, compensation_ratio)),
    }


def neuron_ensemble_effect(model, tokens, layer, top_k=5):
    """Compare effect of ablating neurons individually vs together.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        top_k: number of top neurons to test

    Returns:
        dict with ensemble vs individual effects.
    """
    _, cache = model.run_with_cache(tokens)
    clean_logits = model(tokens)
    post_key = f'blocks.{layer}.mlp.hook_post'
    if post_key not in cache:
        return {'error': 'no post-activation found'}
    mlp_post = cache[post_key]

    # Find top-k most active neurons
    mean_acts = jnp.mean(jnp.abs(mlp_post), axis=0)
    top_indices = [int(i) for i in jnp.argsort(-mean_acts)[:top_k]]

    hook_name = f'blocks.{layer}.mlp.hook_post'

    # Individual effects
    individual_effects = []
    for idx in top_indices:
        def make_hook(target_idx):
            def hook_fn(x, name):
                return x.at[:, target_idx].set(0.0)
            return hook_fn

        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(idx))])
        effect = float(jnp.mean(jnp.abs(mod_logits - clean_logits)))
        individual_effects.append({'neuron_idx': idx, 'individual_effect': effect})

    # Joint effect
    def zero_all(x, name):
        for idx in top_indices:
            x = x.at[:, idx].set(0.0)
        return x

    joint_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_all)])
    joint_effect = float(jnp.mean(jnp.abs(joint_logits - clean_logits)))
    sum_individual = sum(e['individual_effect'] for e in individual_effects)

    # Superadditivity: joint > sum(individual) means synergy
    # Subadditivity: joint < sum(individual) means redundancy
    synergy_ratio = joint_effect / max(sum_individual, 1e-10)

    return {
        'individual_effects': individual_effects,
        'sum_individual': sum_individual,
        'joint_effect': joint_effect,
        'synergy_ratio': synergy_ratio,
        'is_synergistic': synergy_ratio > 1.05,
        'is_redundant': synergy_ratio < 0.95,
    }
