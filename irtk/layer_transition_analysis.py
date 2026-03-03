"""Layer transition analysis: how representations change between layers.

Tools for understanding what each layer does to the residual stream:
- Layer-to-layer representation change
- Component contribution to transition
- Transition smoothness/sharpness
- Layer identity detection (near-identity layers)
- Critical transition localization
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_transition_magnitude(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how much each layer changes the residual stream.

    Computes the norm of each layer's total contribution (attn + mlp)
    relative to the residual stream norm.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer transition magnitudes.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid_pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        resid_post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])

        delta = resid_post - resid_pre
        delta_norm = float(np.linalg.norm(delta))
        pre_norm = float(np.linalg.norm(resid_pre))

        relative_change = delta_norm / pre_norm if pre_norm > 1e-10 else 0.0

        # Cosine similarity between pre and post (direction preservation)
        cos_sim = float(np.dot(resid_pre, resid_post) / (
            pre_norm * float(np.linalg.norm(resid_post)) + 1e-10
        ))

        per_layer.append({
            'layer': l,
            'delta_norm': round(delta_norm, 4),
            'relative_change': round(relative_change, 4),
            'direction_preservation': round(cos_sim, 4),
        })

    return {
        'per_layer': per_layer,
        'max_change_layer': max(per_layer, key=lambda x: x['relative_change'])['layer'],
        'mean_relative_change': round(float(np.mean([p['relative_change'] for p in per_layer])), 4),
    }


def component_transition_contribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Break down each layer's transition into attention vs MLP contributions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer attn/MLP contribution breakdown.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        attn_out = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp_out = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])

        attn_norm = float(np.linalg.norm(attn_out))
        mlp_norm = float(np.linalg.norm(mlp_out))
        total_norm = attn_norm + mlp_norm

        # Cosine between attn and mlp (cooperation vs opposition)
        if attn_norm > 1e-10 and mlp_norm > 1e-10:
            cos = float(np.dot(attn_out, mlp_out) / (attn_norm * mlp_norm))
        else:
            cos = 0.0

        per_layer.append({
            'layer': l,
            'attn_norm': round(attn_norm, 4),
            'mlp_norm': round(mlp_norm, 4),
            'attn_fraction': round(attn_norm / total_norm, 4) if total_norm > 0 else 0.0,
            'mlp_fraction': round(mlp_norm / total_norm, 4) if total_norm > 0 else 0.0,
            'attn_mlp_cosine': round(cos, 4),
            'cooperative': cos > 0,
        })

    return {
        'per_layer': per_layer,
        'attn_dominant_layers': [p['layer'] for p in per_layer if p['attn_fraction'] > 0.6],
        'mlp_dominant_layers': [p['layer'] for p in per_layer if p['mlp_fraction'] > 0.6],
    }


def transition_smoothness(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how smooth or abrupt layer transitions are.

    Smooth transitions have similar deltas in adjacent layers.
    Abrupt transitions have very different deltas.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with smoothness metrics.
    """
    _, cache = model.run_with_cache(tokens)

    deltas = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        deltas.append(post - pre)

    transitions = []
    for i in range(1, len(deltas)):
        ni = float(np.linalg.norm(deltas[i - 1]))
        nj = float(np.linalg.norm(deltas[i]))
        if ni > 1e-10 and nj > 1e-10:
            cos = float(np.dot(deltas[i - 1], deltas[i]) / (ni * nj))
        else:
            cos = 0.0

        norm_ratio = min(ni, nj) / max(ni, nj) if max(ni, nj) > 1e-10 else 1.0

        transitions.append({
            'from_layer': i - 1,
            'to_layer': i,
            'direction_similarity': round(cos, 4),
            'magnitude_ratio': round(norm_ratio, 4),
            'smooth': cos > 0.3 and norm_ratio > 0.5,
        })

    n_smooth = sum(1 for t in transitions if t['smooth'])
    return {
        'transitions': transitions,
        'n_smooth': n_smooth,
        'n_abrupt': len(transitions) - n_smooth,
        'smoothness_score': round(n_smooth / len(transitions), 4) if transitions else 1.0,
    }


def identity_layers(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    threshold: float = 0.1,
) -> dict:
    """Find layers that act approximately as identity (minimal change).

    Near-identity layers suggest the model has more depth than needed
    for this input.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        threshold: Maximum relative change to count as identity.

    Returns:
        Dict with identity layer detection.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid_pre = np.array(cache[f'blocks.{l}.hook_resid_pre'])
        resid_post = np.array(cache[f'blocks.{l}.hook_resid_post'])

        delta = resid_post - resid_pre
        # Average over positions
        relative_changes = []
        for p in range(seq_len):
            pre_norm = float(np.linalg.norm(resid_pre[p]))
            delta_norm = float(np.linalg.norm(delta[p]))
            if pre_norm > 1e-10:
                relative_changes.append(delta_norm / pre_norm)

        mean_change = float(np.mean(relative_changes)) if relative_changes else 0.0
        is_identity = mean_change < threshold

        per_layer.append({
            'layer': l,
            'mean_relative_change': round(mean_change, 4),
            'is_identity': is_identity,
        })

    identity = [p['layer'] for p in per_layer if p['is_identity']]
    return {
        'per_layer': per_layer,
        'identity_layers': identity,
        'n_identity': len(identity),
        'threshold': threshold,
    }


def critical_transitions(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Find layers where the prediction changes most.

    Measures each layer's effect on the logit of the target token.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to track (default: top prediction).

    Returns:
        Dict with per-layer logit changes.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U[:, target_token])  # [d_model]

    per_layer = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])

        logit_pre = float(np.dot(pre, W_U))
        logit_post = float(np.dot(post, W_U))
        logit_delta = logit_post - logit_pre

        # Break down into attn/MLP
        attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])
        attn_logit = float(np.dot(attn, W_U))
        mlp_logit = float(np.dot(mlp, W_U))

        per_layer.append({
            'layer': l,
            'logit_before': round(logit_pre, 4),
            'logit_after': round(logit_post, 4),
            'logit_delta': round(logit_delta, 4),
            'attn_logit_delta': round(attn_logit, 4),
            'mlp_logit_delta': round(mlp_logit, 4),
        })

    # Most critical = largest absolute delta
    sorted_layers = sorted(per_layer, key=lambda x: -abs(x['logit_delta']))
    return {
        'target_token': target_token,
        'per_layer': per_layer,
        'most_critical_layer': sorted_layers[0]['layer'] if sorted_layers else 0,
    }
