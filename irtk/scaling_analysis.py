"""Scaling analysis: how model properties change with depth and width.

Tools for understanding how interpretability-relevant properties scale:
- Layer-wise capacity utilization
- Effective rank growth across layers
- Feature density estimation
- Component saturation detection
- Width-depth tradeoff analysis
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_capacity_utilization(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
) -> dict:
    """Measure how much of each layer's capacity is being used.

    Computes effective rank (via singular value entropy) of each layer's
    output across positions, showing how many dimensions are actively used.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per_layer results and summary statistics.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        resid = cache[f'blocks.{l}.hook_resid_post']  # [seq, d_model]
        # SVD of the residual at this layer
        U, s, Vt = jnp.linalg.svd(resid, full_matrices=False)
        s = jnp.maximum(s, 1e-10)
        # Normalized singular values
        s_norm = s / jnp.sum(s)
        # Effective rank via exponential of entropy
        entropy = -float(jnp.sum(s_norm * jnp.log(s_norm)))
        effective_rank = float(jnp.exp(entropy))
        max_rank = min(resid.shape[0], resid.shape[1])
        utilization = effective_rank / max_rank

        # Top singular value dominance
        top1_frac = float(s[0] / jnp.sum(s))

        per_layer.append({
            'layer': l,
            'effective_rank': round(effective_rank, 2),
            'max_rank': max_rank,
            'utilization': round(utilization, 4),
            'top1_fraction': round(top1_frac, 4),
            'singular_values': np.array(s[:min(10, len(s))]).tolist(),
        })

    ranks = [p['effective_rank'] for p in per_layer]
    return {
        'per_layer': per_layer,
        'mean_effective_rank': round(float(np.mean(ranks)), 2),
        'rank_growth': round(ranks[-1] - ranks[0], 2) if len(ranks) > 1 else 0.0,
    }


def feature_density(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    layer: int = -1,
    pos: int = -1,
    n_directions: int = 50,
    seed: int = 42,
) -> dict:
    """Estimate feature density in the residual stream at a given point.

    Projects activations onto random directions and measures how many
    show bimodal (feature-like) distributions, suggesting active features.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        layer: Layer to analyze (-1 for last).
        pos: Position to analyze (-1 for last).
        n_directions: Number of random directions to probe.
        seed: Random seed.

    Returns:
        Dict with density estimate and per-direction statistics.
    """
    if layer < 0:
        layer = model.cfg.n_layers + layer

    hook_name = f'blocks.{layer}.hook_resid_post'
    activations = []
    for tokens in tokens_list:
        _, cache = model.run_with_cache(tokens)
        activations.append(np.array(cache[hook_name][pos]))

    acts = np.stack(activations)  # [n_examples, d_model]
    n_examples = acts.shape[0]

    # Generate random directions
    rng = np.random.RandomState(seed)
    directions = rng.randn(n_directions, model.cfg.d_model)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # Project onto each direction and check for bimodality
    n_bimodal = 0
    per_direction = []
    for i in range(n_directions):
        projections = acts @ directions[i]  # [n_examples]
        mean = float(np.mean(projections))
        std = float(np.std(projections))
        # Simple bimodality test: kurtosis < 3 suggests bimodal
        if std > 1e-10:
            centered = (projections - mean) / std
            kurtosis = float(np.mean(centered ** 4))
        else:
            kurtosis = 3.0

        is_bimodal = kurtosis < 2.5
        if is_bimodal:
            n_bimodal += 1

        per_direction.append({
            'direction': i,
            'mean': round(mean, 4),
            'std': round(std, 4),
            'kurtosis': round(kurtosis, 4),
            'is_bimodal': is_bimodal,
        })

    density = n_bimodal / n_directions
    return {
        'density_estimate': round(density, 4),
        'n_bimodal': n_bimodal,
        'n_directions': n_directions,
        'layer': layer,
        'per_direction': per_direction[:10],  # First 10 for brevity
    }


def component_saturation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
) -> dict:
    """Detect whether components are saturated (producing near-max outputs).

    Checks if attention patterns are near-one-hot (saturated softmax) and
    if MLP activations are near their max/min (saturated nonlinearity).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with saturation metrics per layer.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        # Attention saturation: how peaked are the attention patterns?
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]
        max_attn = np.array(jnp.max(pattern, axis=-1))  # [n_heads, seq]
        mean_max = float(np.mean(max_attn))
        n_saturated_heads = int(np.sum(np.mean(max_attn, axis=1) > 0.9))

        # MLP saturation: how many neurons are near 0 (dead) or very large?
        post = cache[f'blocks.{l}.mlp.hook_post']  # [seq, d_mlp]
        post_np = np.array(post)
        dead_frac = float(np.mean(np.abs(post_np) < 0.01))
        max_act = float(np.max(np.abs(post_np)))

        per_layer.append({
            'layer': l,
            'attn_mean_max': round(mean_max, 4),
            'attn_saturated_heads': n_saturated_heads,
            'mlp_dead_fraction': round(dead_frac, 4),
            'mlp_max_activation': round(max_act, 4),
        })

    return {
        'per_layer': per_layer,
        'any_attn_saturated': any(p['attn_saturated_heads'] > 0 for p in per_layer),
        'mean_dead_fraction': round(float(np.mean([p['mlp_dead_fraction'] for p in per_layer])), 4),
    }


def depth_contribution_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Analyze how each layer's contribution magnitude scales with depth.

    Measures the norm of each component's output relative to the residual
    stream, showing which layers contribute most.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer contribution norms and ratios.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid_pre = cache[f'blocks.{l}.hook_resid_pre'][pos]
        attn_out = cache[f'blocks.{l}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{l}.hook_mlp_out'][pos]

        resid_norm = float(jnp.linalg.norm(resid_pre))
        attn_norm = float(jnp.linalg.norm(attn_out))
        mlp_norm = float(jnp.linalg.norm(mlp_out))

        attn_ratio = attn_norm / resid_norm if resid_norm > 1e-10 else 0.0
        mlp_ratio = mlp_norm / resid_norm if resid_norm > 1e-10 else 0.0

        per_layer.append({
            'layer': l,
            'residual_norm': round(resid_norm, 4),
            'attn_contribution_norm': round(attn_norm, 4),
            'mlp_contribution_norm': round(mlp_norm, 4),
            'attn_to_residual_ratio': round(attn_ratio, 4),
            'mlp_to_residual_ratio': round(mlp_ratio, 4),
        })

    return {
        'per_layer': per_layer,
        'total_attn_contribution': round(sum(p['attn_contribution_norm'] for p in per_layer), 4),
        'total_mlp_contribution': round(sum(p['mlp_contribution_norm'] for p in per_layer), 4),
    }


def representation_compression(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    layers: Optional[list[int]] = None,
    pos: int = -1,
) -> dict:
    """Measure how compressed/spread representations are at each layer.

    Computes the effective dimensionality of the point cloud of activations
    from different inputs, showing how the model uses its capacity.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        layers: Layers to analyze (default: all).
        pos: Position to analyze.

    Returns:
        Dict with compression metrics per layer.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        hook_name = f'blocks.{l}.hook_resid_post'
        acts = []
        for tokens in tokens_list:
            _, cache = model.run_with_cache(tokens)
            acts.append(np.array(cache[hook_name][pos]))

        acts_mat = np.stack(acts)  # [n_examples, d_model]
        # Center
        acts_centered = acts_mat - acts_mat.mean(axis=0)

        # SVD for effective dimensionality
        if acts_centered.shape[0] > 1:
            U, s, Vt = np.linalg.svd(acts_centered, full_matrices=False)
            s = np.maximum(s, 1e-10)
            s_norm = s / s.sum()
            entropy = -float(np.sum(s_norm * np.log(s_norm)))
            eff_dim = float(np.exp(entropy))
        else:
            eff_dim = 1.0

        # Mean pairwise distance
        dists = []
        for i in range(min(len(acts_mat), 20)):
            for j in range(i + 1, min(len(acts_mat), 20)):
                d = float(np.linalg.norm(acts_mat[i] - acts_mat[j]))
                dists.append(d)
        mean_dist = float(np.mean(dists)) if dists else 0.0

        per_layer.append({
            'layer': l,
            'effective_dimensionality': round(eff_dim, 2),
            'mean_pairwise_distance': round(mean_dist, 4),
        })

    dims = [p['effective_dimensionality'] for p in per_layer]
    return {
        'per_layer': per_layer,
        'compression_trend': round(dims[-1] - dims[0], 2) if len(dims) > 1 else 0.0,
    }
