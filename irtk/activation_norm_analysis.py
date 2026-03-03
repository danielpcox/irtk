"""Activation norm analysis: tracking activation magnitudes through the model.

Tools for understanding activation norms and their dynamics:
- Per-layer residual norm tracking
- Component output norm comparison
- Norm growth/decay patterns
- Norm concentration analysis
- Pre/post LayerNorm norm effects
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def residual_norm_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Track the residual stream norm through each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with per-layer, per-position residual norms.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    per_layer = []

    # Embedding
    resid_0 = np.array(cache['blocks.0.hook_resid_pre'])
    norms_0 = np.linalg.norm(resid_0, axis=1)
    per_layer.append({
        'stage': 'embedding',
        'mean_norm': round(float(np.mean(norms_0)), 4),
        'max_norm': round(float(np.max(norms_0)), 4),
        'min_norm': round(float(np.min(norms_0)), 4),
    })

    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'])
        norms = np.linalg.norm(resid, axis=1)
        per_layer.append({
            'stage': f'layer_{l}',
            'mean_norm': round(float(np.mean(norms)), 4),
            'max_norm': round(float(np.max(norms)), 4),
            'min_norm': round(float(np.min(norms)), 4),
        })

    # Growth pattern
    norms_seq = [p['mean_norm'] for p in per_layer]
    growth_rates = []
    for i in range(1, len(norms_seq)):
        if norms_seq[i-1] > 0:
            growth_rates.append(norms_seq[i] / norms_seq[i-1])

    return {
        'per_layer': per_layer,
        'overall_growth': round(norms_seq[-1] / norms_seq[0], 4) if norms_seq[0] > 0 else 0.0,
        'mean_growth_rate': round(float(np.mean(growth_rates)), 4) if growth_rates else 1.0,
    }


def component_norm_comparison(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Compare output norms of attention vs MLP at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer component norm comparison.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])

        attn_norm = float(np.linalg.norm(attn))
        mlp_norm = float(np.linalg.norm(mlp))
        resid_norm = float(np.linalg.norm(resid))

        per_layer.append({
            'layer': l,
            'attn_norm': round(attn_norm, 4),
            'mlp_norm': round(mlp_norm, 4),
            'residual_norm': round(resid_norm, 4),
            'attn_fraction': round(attn_norm / (attn_norm + mlp_norm), 4) if (attn_norm + mlp_norm) > 0 else 0.0,
        })

    return {'per_layer': per_layer}


def norm_concentration(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how concentrated the residual norm is across dimensions.

    High concentration means a few dimensions dominate the norm.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer norm concentration.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        sq = resid ** 2
        total = float(np.sum(sq))

        if total > 0:
            fractions = sq / total
            # Entropy of squared values
            entropy = -float(np.sum(fractions * np.log(fractions + 1e-10)))
            max_entropy = float(np.log(len(resid)))
            concentration = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0

            # Top-k dimensions
            sorted_dims = np.argsort(sq)[::-1]
            top5_fraction = float(np.sum(sq[sorted_dims[:5]]) / total)
        else:
            concentration = 0.0
            top5_fraction = 0.0

        per_layer.append({
            'layer': l,
            'concentration': round(concentration, 4),
            'top5_norm_fraction': round(top5_fraction, 4),
            'total_norm_squared': round(total, 4),
        })

    return {'per_layer': per_layer}


def position_norm_variation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Measure how residual norms vary across positions at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with per-layer norm variation across positions.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'])
        norms = np.linalg.norm(resid, axis=1)

        per_layer.append({
            'layer': l,
            'norms': [round(float(n), 4) for n in norms],
            'std': round(float(np.std(norms)), 4),
            'cv': round(float(np.std(norms) / np.mean(norms)), 4) if float(np.mean(norms)) > 0 else 0.0,
            'max_min_ratio': round(float(np.max(norms) / np.min(norms)), 4) if float(np.min(norms)) > 0 else 0.0,
        })

    return {'per_layer': per_layer}


def norm_growth_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Attribute norm growth to attention vs MLP at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer growth attribution.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])

        pre_norm_sq = float(np.sum(pre ** 2))
        post_norm_sq = float(np.sum(post ** 2))
        norm_change = post_norm_sq - pre_norm_sq

        # Decompose: ||pre + attn + mlp||^2 = ||pre||^2 + ||attn||^2 + ||mlp||^2
        #   + 2*<pre, attn> + 2*<pre, mlp> + 2*<attn, mlp>
        attn_self = float(np.sum(attn ** 2))
        mlp_self = float(np.sum(mlp ** 2))
        pre_attn = 2 * float(np.dot(pre, attn))
        pre_mlp = 2 * float(np.dot(pre, mlp))
        attn_mlp = 2 * float(np.dot(attn, mlp))

        per_layer.append({
            'layer': l,
            'norm_change': round(norm_change, 4),
            'attn_self_contribution': round(attn_self, 4),
            'mlp_self_contribution': round(mlp_self, 4),
            'pre_attn_interaction': round(pre_attn, 4),
            'pre_mlp_interaction': round(pre_mlp, 4),
            'attn_mlp_interaction': round(attn_mlp, 4),
        })

    return {'per_layer': per_layer}
