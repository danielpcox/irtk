"""Feature universality analysis.

Test whether features are universal across different inputs, positions,
and contexts: feature activation consistency, position independence,
context invariance, and feature clustering across inputs.
"""

import jax
import jax.numpy as jnp


def feature_activation_consistency(model, tokens_list, layer, direction):
    """Test if a feature direction activates consistently across different inputs.

    Args:
        model: HookedTransformer
        tokens_list: list of different token arrays
        layer: layer to analyze
        direction: feature direction [d_model]

    Returns:
        dict with consistency analysis.
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    per_input = []
    for i, tokens in enumerate(tokens_list):
        _, cache = model.run_with_cache(tokens)
        resid = cache[f'blocks.{layer}.hook_resid_post']
        projections = resid @ direction  # [seq]
        mean_proj = float(jnp.mean(projections))
        max_proj = float(jnp.max(projections))
        active_frac = float(jnp.mean(projections > 0))

        per_input.append({
            'input_idx': i,
            'mean_projection': mean_proj,
            'max_projection': max_proj,
            'active_fraction': active_frac,
        })

    mean_projs = [p['mean_projection'] for p in per_input]
    consistency = 1.0 - float(jnp.std(jnp.array(mean_projs))) / (abs(float(jnp.mean(jnp.array(mean_projs)))) + 1e-10)

    return {
        'per_input': per_input,
        'consistency_score': max(0.0, min(1.0, consistency)),
        'mean_across_inputs': float(jnp.mean(jnp.array(mean_projs))),
    }


def position_independence(model, tokens, layer, direction):
    """Test if a feature activates independently of position.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer to analyze
        direction: feature direction [d_model]

    Returns:
        dict with position independence analysis.
    """
    _, cache = model.run_with_cache(tokens)
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)
    resid = cache[f'blocks.{layer}.hook_resid_post']
    projections = resid @ direction  # [seq]

    seq_len = len(tokens)
    positions = jnp.arange(seq_len).astype(jnp.float32)

    # Correlation between position and projection
    proj_centered = projections - jnp.mean(projections)
    pos_centered = positions - jnp.mean(positions)
    corr_num = jnp.sum(proj_centered * pos_centered)
    corr_denom = jnp.sqrt(jnp.sum(proj_centered**2) * jnp.sum(pos_centered**2) + 1e-10)
    pos_correlation = float(corr_num / corr_denom)

    per_position = []
    for pos in range(seq_len):
        per_position.append({
            'position': pos,
            'projection': float(projections[pos]),
        })

    return {
        'position_correlation': pos_correlation,
        'is_position_independent': abs(pos_correlation) < 0.3,
        'per_position': per_position,
        'projection_std': float(jnp.std(projections)),
    }


def context_invariance(model, tokens_a, tokens_b, layer, direction):
    """Test if a feature responds similarly in different contexts.

    Args:
        model: HookedTransformer
        tokens_a: first context
        tokens_b: second context
        layer: layer to analyze
        direction: feature direction [d_model]

    Returns:
        dict with context invariance.
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)

    resid_a = cache_a[f'blocks.{layer}.hook_resid_post']
    resid_b = cache_b[f'blocks.{layer}.hook_resid_post']

    proj_a = resid_a @ direction
    proj_b = resid_b @ direction

    mean_a = float(jnp.mean(proj_a))
    mean_b = float(jnp.mean(proj_b))
    diff = abs(mean_a - mean_b)
    scale = max(abs(mean_a), abs(mean_b), 1e-10)

    # Distribution comparison
    std_a = float(jnp.std(proj_a))
    std_b = float(jnp.std(proj_b))

    return {
        'mean_projection_a': mean_a,
        'mean_projection_b': mean_b,
        'absolute_difference': diff,
        'relative_difference': diff / scale,
        'std_a': std_a,
        'std_b': std_b,
        'is_invariant': diff / scale < 0.3,
    }


def feature_clustering_across_inputs(model, tokens_list, layer, n_directions=5):
    """Find shared principal directions across multiple inputs.

    Args:
        model: HookedTransformer
        tokens_list: list of different token arrays
        layer: layer to analyze
        n_directions: number of principal directions

    Returns:
        dict with shared feature analysis.
    """
    all_resids = []
    for tokens in tokens_list:
        _, cache = model.run_with_cache(tokens)
        resid = cache[f'blocks.{layer}.hook_resid_post']
        all_resids.append(resid)

    # Stack all residuals
    combined = jnp.concatenate(all_resids, axis=0)

    # PCA via SVD
    centered = combined - jnp.mean(combined, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
    directions = Vt[:n_directions]  # [n_directions, d_model]

    # Check how much variance each direction explains per input
    per_direction = []
    total_var = float(jnp.sum(S**2))
    for d in range(n_directions):
        explained = float(S[d]**2) / (total_var + 1e-10)

        # Consistency: does this direction explain similar variance in each input?
        per_input_var = []
        for resid in all_resids:
            proj = resid @ directions[d]
            per_input_var.append(float(jnp.var(proj)))

        var_consistency = 1.0 - float(jnp.std(jnp.array(per_input_var))) / (float(jnp.mean(jnp.array(per_input_var))) + 1e-10)

        per_direction.append({
            'direction_idx': d,
            'variance_explained': explained,
            'per_input_variance': per_input_var,
            'consistency': max(0.0, min(1.0, var_consistency)),
        })

    return {
        'per_direction': per_direction,
        'n_inputs': len(tokens_list),
        'total_variance': total_var,
        'mean_consistency': float(jnp.mean(jnp.array([d['consistency'] for d in per_direction]))),
    }


def layer_wise_feature_universality(model, tokens_list, direction):
    """Test feature universality across all layers.

    Args:
        model: HookedTransformer
        tokens_list: list of different token arrays
        direction: feature direction [d_model]

    Returns:
        dict with per-layer universality scores.
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)
    n_layers = model.cfg.n_layers

    per_layer = []
    for l in range(n_layers):
        mean_projs = []
        for tokens in tokens_list:
            _, cache = model.run_with_cache(tokens)
            resid = cache[f'blocks.{l}.hook_resid_post']
            mean_projs.append(float(jnp.mean(resid @ direction)))

        mean_projs_arr = jnp.array(mean_projs)
        mean_val = float(jnp.mean(mean_projs_arr))
        std_val = float(jnp.std(mean_projs_arr))
        cv = std_val / (abs(mean_val) + 1e-10)

        per_layer.append({
            'layer': l,
            'mean_projection': mean_val,
            'std_projection': std_val,
            'coefficient_of_variation': cv,
            'is_universal': cv < 0.5,
        })

    n_universal = sum(1 for p in per_layer if p['is_universal'])
    return {
        'per_layer': per_layer,
        'universal_fraction': n_universal / max(n_layers, 1),
        'most_universal_layer': min(per_layer, key=lambda p: p['coefficient_of_variation'])['layer'] if per_layer else 0,
    }
