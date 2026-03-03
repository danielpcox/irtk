"""MLP output decomposition.

Decompose MLP layer outputs into interpretable parts: per-neuron output
direction, top contributing neurons per token, neuron selectivity profiles,
and output direction clustering.
"""

import jax
import jax.numpy as jnp


def per_neuron_output_direction(model, tokens, layer, top_k=10):
    """Analyze the output direction of the most active neurons.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        top_k: number of top neurons

    Returns:
        dict with per-neuron output analysis.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Mean activation per neuron
    mean_acts = jnp.mean(jnp.abs(mlp_post), axis=0)  # [d_mlp]
    top_indices = jnp.argsort(-mean_acts)[:top_k]

    per_neuron = []
    for idx in top_indices:
        idx = int(idx)
        direction = W_out[idx]  # [d_model]
        dir_norm = float(jnp.linalg.norm(direction))

        # What vocabulary does this neuron promote?
        logits = direction @ W_U  # [d_vocab]
        top_token = int(jnp.argmax(logits))
        bottom_token = int(jnp.argmin(logits))

        per_neuron.append({
            'neuron_idx': idx,
            'mean_activation': float(mean_acts[idx]),
            'output_direction_norm': dir_norm,
            'top_promoted_token': top_token,
            'top_suppressed_token': bottom_token,
            'top_logit': float(logits[top_token]),
        })

    return {
        'layer': layer,
        'per_neuron': per_neuron,
        'n_analyzed': len(per_neuron),
    }


def position_neuron_contributions(model, tokens, layer, position=-1, top_k=5):
    """Find which neurons contribute most at a specific position.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        position: position to analyze
        top_k: number of top neurons

    Returns:
        dict with per-neuron contribution at position.
    """
    _, cache = model.run_with_cache(tokens)
    pos = position if position >= 0 else len(tokens) - 1
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    activations = mlp_post[pos]  # [d_mlp]
    contributions = jnp.abs(activations) * jnp.linalg.norm(W_out, axis=-1)  # [d_mlp]

    top_indices = jnp.argsort(-contributions)[:top_k]
    total_contribution = float(jnp.sum(contributions))

    per_neuron = []
    for idx in top_indices:
        idx = int(idx)
        per_neuron.append({
            'neuron_idx': idx,
            'activation': float(activations[idx]),
            'contribution': float(contributions[idx]),
            'fraction': float(contributions[idx]) / max(total_contribution, 1e-10),
        })

    return {
        'layer': layer,
        'position': pos,
        'per_neuron': per_neuron,
        'total_contribution': total_contribution,
    }


def neuron_selectivity_profile(model, tokens, layer, top_k=10):
    """Profile neuron selectivity: which neurons are position-selective vs uniform.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        top_k: number of neurons to profile

    Returns:
        dict with selectivity profiles.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]

    mean_acts = jnp.mean(jnp.abs(mlp_post), axis=0)
    top_indices = jnp.argsort(-mean_acts)[:top_k]

    per_neuron = []
    for idx in top_indices:
        idx = int(idx)
        acts = mlp_post[:, idx]  # [seq]
        abs_acts = jnp.abs(acts)

        # Selectivity: how concentrated across positions?
        total = float(jnp.sum(abs_acts))
        if total > 1e-10:
            probs = abs_acts / total
            entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
            max_entropy = float(jnp.log(jnp.array(len(tokens))))
            selectivity = 1.0 - entropy / (max_entropy + 1e-10)
        else:
            selectivity = 0.0

        max_pos = int(jnp.argmax(abs_acts))
        per_neuron.append({
            'neuron_idx': idx,
            'mean_activation': float(mean_acts[idx]),
            'selectivity': selectivity,
            'is_selective': selectivity > 0.5,
            'peak_position': max_pos,
        })

    return {
        'layer': layer,
        'per_neuron': per_neuron,
        'n_selective': sum(1 for n in per_neuron if n['is_selective']),
    }


def mlp_output_direction_clustering(model, tokens, layer, n_clusters=3):
    """Cluster neurons by their output direction in W_out.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer
        n_clusters: number of clusters

    Returns:
        dict with direction clustering.
    """
    _, cache = model.run_with_cache(tokens)
    mlp_post = cache[f'blocks.{layer}.mlp.hook_post']  # [seq, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    # Only consider active neurons
    mean_acts = jnp.mean(jnp.abs(mlp_post), axis=0)
    active_mask = mean_acts > float(jnp.mean(mean_acts))
    active_indices = jnp.where(active_mask)[0]

    if len(active_indices) < n_clusters:
        return {'layer': layer, 'clusters': [], 'n_active': int(len(active_indices))}

    # Normalize directions
    directions = W_out[active_indices]  # [n_active, d_model]
    norms = jnp.linalg.norm(directions, axis=-1, keepdims=True) + 1e-10
    normalized = directions / norms

    # Simple k-means
    key = jax.random.PRNGKey(0)
    centers = normalized[jax.random.choice(key, len(active_indices), (n_clusters,), replace=False)]

    for _ in range(10):
        # Assign
        sims = normalized @ centers.T  # [n_active, n_clusters]
        assignments = jnp.argmax(sims, axis=-1)

        # Update centers
        new_centers = []
        for c in range(n_clusters):
            mask = assignments == c
            if jnp.any(mask):
                center = jnp.mean(normalized[mask], axis=0)
                center = center / (jnp.linalg.norm(center) + 1e-10)
                new_centers.append(center)
            else:
                new_centers.append(centers[c])
        centers = jnp.stack(new_centers)

    clusters = []
    for c in range(n_clusters):
        mask = assignments == c
        n_members = int(jnp.sum(mask))
        mean_act = float(jnp.mean(mean_acts[active_indices[mask]])) if n_members > 0 else 0.0
        clusters.append({
            'cluster': c,
            'n_members': n_members,
            'mean_activation': mean_act,
        })

    return {
        'layer': layer,
        'clusters': clusters,
        'n_active': int(len(active_indices)),
        'n_clusters': n_clusters,
    }


def mlp_residual_alignment(model, tokens, layer):
    """Measure how well MLP output aligns with the existing residual stream.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: MLP layer

    Returns:
        dict with alignment analysis.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[f'blocks.{layer}.hook_resid_pre']  # [seq, d_model]
    mlp_out = cache[f'blocks.{layer}.hook_mlp_out']  # [seq, d_model]

    per_position = []
    for pos in range(len(tokens)):
        r = resid_pre[pos]
        m = mlp_out[pos]
        r_norm = jnp.linalg.norm(r)
        m_norm = jnp.linalg.norm(m)
        cos = float(jnp.sum(r * m) / (r_norm * m_norm + 1e-10))

        per_position.append({
            'position': pos,
            'cosine_alignment': cos,
            'residual_norm': float(r_norm),
            'mlp_output_norm': float(m_norm),
            'reinforces': cos > 0,
        })

    alignments = [p['cosine_alignment'] for p in per_position]
    return {
        'layer': layer,
        'per_position': per_position,
        'mean_alignment': float(jnp.mean(jnp.array(alignments))),
        'reinforcement_fraction': sum(1 for p in per_position if p['reinforces']) / max(len(per_position), 1),
    }
