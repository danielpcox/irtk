"""Residual stream bottleneck analysis.

Tools for finding information bottlenecks in the residual stream:
- Dimension utilization across layers
- Information compression points
- Redundancy detection
- Capacity allocation analysis
- Critical dimension identification
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def dimension_utilization(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
    threshold: float = 0.01,
) -> dict:
    """Measure how many residual stream dimensions are actively used per layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).
        threshold: Fraction of max variance to count a dimension as active.

    Returns:
        Dict with per-layer dimension utilization.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        resid = cache[f'blocks.{l}.hook_resid_post']  # [seq, d_model]

        # Variance along each dimension across positions
        var_per_dim = np.array(jnp.var(resid, axis=0))  # [d_model]
        max_var = float(np.max(var_per_dim))

        if max_var > 1e-10:
            n_active = int(np.sum(var_per_dim > threshold * max_var))
        else:
            n_active = 0

        per_layer.append({
            'layer': l,
            'active_dimensions': n_active,
            'total_dimensions': model.cfg.d_model,
            'utilization': round(n_active / model.cfg.d_model, 4),
            'max_variance': round(max_var, 6),
        })

    return {
        'per_layer': per_layer,
        'mean_utilization': round(float(np.mean([p['utilization'] for p in per_layer])), 4),
    }


def compression_points(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Find layers where the representation compresses (loses dimensions).

    Compares effective rank before and after each layer to detect
    compression bottlenecks.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer compression ratios.
    """
    _, cache = model.run_with_cache(tokens)

    ranks = []
    for l in range(model.cfg.n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']  # [seq, d_model]
        # SVD on the full sequence
        U, s, Vt = jnp.linalg.svd(resid, full_matrices=False)
        s = jnp.maximum(s, 1e-10)
        s_norm = s / jnp.sum(s)
        entropy = -float(jnp.sum(s_norm * jnp.log(s_norm)))
        eff_rank = float(jnp.exp(entropy))
        ranks.append(eff_rank)

    per_layer = []
    bottlenecks = []
    for l in range(len(ranks)):
        if l > 0:
            ratio = ranks[l] / ranks[l - 1] if ranks[l - 1] > 0 else 1.0
        else:
            ratio = 1.0

        is_bottleneck = ratio < 0.8  # Significant compression
        per_layer.append({
            'layer': l,
            'effective_rank': round(ranks[l], 2),
            'compression_ratio': round(ratio, 4),
            'is_bottleneck': is_bottleneck,
        })
        if is_bottleneck:
            bottlenecks.append(l)

    return {
        'per_layer': per_layer,
        'bottleneck_layers': bottlenecks,
        'n_bottlenecks': len(bottlenecks),
    }


def redundancy_detection(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
    pos: int = -1,
) -> dict:
    """Detect redundant dimensions in the residual stream.

    Finds dimensions that carry highly correlated information,
    suggesting redundant capacity usage.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).
        pos: Position to analyze.

    Returns:
        Dict with redundancy metrics per layer.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        resid = cache[f'blocks.{l}.hook_resid_post']  # [seq, d_model]
        resid_np = np.array(resid)

        # Correlation matrix between dimensions
        # Center first
        centered = resid_np - resid_np.mean(axis=0)
        norms = np.linalg.norm(centered, axis=0)
        norms = np.maximum(norms, 1e-10)
        normalized = centered / norms

        corr = normalized.T @ normalized / centered.shape[0]  # [d_model, d_model]

        # Count highly correlated pairs (excluding diagonal)
        np.fill_diagonal(corr, 0)
        high_corr = np.abs(corr) > 0.8
        n_redundant_pairs = int(np.sum(high_corr)) // 2  # Each pair counted twice
        max_corr = float(np.max(np.abs(corr)))

        per_layer.append({
            'layer': l,
            'n_redundant_pairs': n_redundant_pairs,
            'max_correlation': round(max_corr, 4),
            'mean_abs_correlation': round(float(np.mean(np.abs(corr))), 4),
        })

    return {
        'per_layer': per_layer,
        'total_redundant_pairs': sum(p['n_redundant_pairs'] for p in per_layer),
    }


def capacity_allocation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Analyze how residual stream capacity is allocated to different components.

    Measures how much of the total variance is explained by each component's
    output direction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-component capacity allocation.
    """
    _, cache = model.run_with_cache(tokens)

    # Get the final residual
    final = cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'][pos]
    total_norm_sq = float(jnp.sum(final ** 2))

    components = []

    # Embedding
    embed = cache['blocks.0.hook_resid_pre'][pos]
    embed_norm_sq = float(jnp.sum(embed ** 2))
    components.append({
        'component': 'embed+pos',
        'norm_squared': round(embed_norm_sq, 4),
        'fraction': round(embed_norm_sq / total_norm_sq, 4) if total_norm_sq > 0 else 0.0,
    })

    for l in range(model.cfg.n_layers):
        attn = cache[f'blocks.{l}.hook_attn_out'][pos]
        mlp = cache[f'blocks.{l}.hook_mlp_out'][pos]

        attn_norm_sq = float(jnp.sum(attn ** 2))
        mlp_norm_sq = float(jnp.sum(mlp ** 2))

        components.append({
            'component': f'L{l}_attn',
            'norm_squared': round(attn_norm_sq, 4),
            'fraction': round(attn_norm_sq / total_norm_sq, 4) if total_norm_sq > 0 else 0.0,
        })
        components.append({
            'component': f'L{l}_mlp',
            'norm_squared': round(mlp_norm_sq, 4),
            'fraction': round(mlp_norm_sq / total_norm_sq, 4) if total_norm_sq > 0 else 0.0,
        })

    # Note: fractions don't sum to 1 due to cross-terms (interference)
    total_fraction = sum(c['fraction'] for c in components)

    return {
        'components': components,
        'total_norm_squared': round(total_norm_sq, 4),
        'sum_of_fractions': round(total_fraction, 4),
        'interference': round(1.0 - total_fraction, 4),
    }


def critical_dimensions(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Find the most important dimensions in the residual stream for the prediction.

    Projects the residual through the unembedding and finds which
    dimensions of the residual stream contribute most to the top prediction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of critical dimensions to return.

    Returns:
        Dict with top contributing dimensions and their logit effects.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    top_token = int(jnp.argmax(logits[pos]))
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    target_dir = np.array(W_U[:, top_token])  # [d_model]

    # Final residual
    resid = np.array(cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'][pos])

    # Per-dimension contribution = resid[d] * target_dir[d]
    contributions = resid * target_dir  # [d_model]

    top_dims = np.argsort(np.abs(contributions))[::-1][:top_k]

    per_dimension = []
    for dim in top_dims:
        per_dimension.append({
            'dimension': int(dim),
            'residual_value': round(float(resid[dim]), 4),
            'unembed_weight': round(float(target_dir[dim]), 4),
            'logit_contribution': round(float(contributions[dim]), 4),
        })

    total_logit = float(np.sum(contributions))
    top_k_logit = sum(d['logit_contribution'] for d in per_dimension)

    return {
        'target_token': top_token,
        'per_dimension': per_dimension,
        'total_logit': round(total_logit, 4),
        'top_k_fraction': round(top_k_logit / total_logit, 4) if abs(total_logit) > 1e-10 else 0.0,
    }
