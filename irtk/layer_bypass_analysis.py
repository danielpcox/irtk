"""Layer bypass analysis: detecting skip connections and shortcuts.

Analyze when and how the model bypasses layers:
- Layer contribution vs pass-through
- Shortcut detection
- Effective depth measurement
- Skip connection utilization
- Minimal circuit depth
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_contribution_vs_passthrough(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure what fraction of each layer's output is new vs passed-through.

    A layer with high pass-through ratio mostly relays the input unchanged.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer contribution vs pass-through ratios.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        delta = post - pre

        pre_norm = float(np.linalg.norm(pre))
        post_norm = float(np.linalg.norm(post))
        delta_norm = float(np.linalg.norm(delta))

        # Pass-through: fraction of output that is the input
        # Contribution: fraction that is new
        if post_norm > 1e-10:
            passthrough = pre_norm / post_norm
            contribution = delta_norm / post_norm
        else:
            passthrough = 1.0
            contribution = 0.0

        per_layer.append({
            'layer': l,
            'passthrough_ratio': round(passthrough, 4),
            'contribution_ratio': round(contribution, 4),
            'is_bypass': contribution < 0.1,
        })

    bypass_layers = [p['layer'] for p in per_layer if p['is_bypass']]
    return {
        'per_layer': per_layer,
        'bypass_layers': bypass_layers,
        'n_bypass': len(bypass_layers),
    }


def effective_depth(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    contribution_threshold: float = 0.05,
) -> dict:
    """Compute the effective depth — how many layers meaningfully contribute.

    Layers below the contribution threshold are considered bypassed.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        contribution_threshold: Minimum relative contribution to count.

    Returns:
        Dict with effective depth metrics.
    """
    _, cache = model.run_with_cache(tokens)

    contributions = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        delta = post - pre
        delta_norm = float(np.linalg.norm(delta))
        pre_norm = float(np.linalg.norm(pre))
        rel_contrib = delta_norm / pre_norm if pre_norm > 1e-10 else 0.0
        contributions.append({
            'layer': l,
            'relative_contribution': round(rel_contrib, 4),
        })

    active_layers = [c for c in contributions if c['relative_contribution'] > contribution_threshold]

    return {
        'total_layers': model.cfg.n_layers,
        'effective_depth': len(active_layers),
        'depth_ratio': round(len(active_layers) / model.cfg.n_layers, 4),
        'per_layer': contributions,
        'active_layers': [c['layer'] for c in active_layers],
        'threshold': contribution_threshold,
    }


def skip_connection_utilization(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how much the model relies on skip connections.

    Compares the embedding's direct contribution to the final output
    vs layer contributions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with skip connection utilization metrics.
    """
    _, cache = model.run_with_cache(tokens)

    embed = np.array(cache['blocks.0.hook_resid_pre'][pos])
    final = np.array(cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'][pos])

    embed_norm = float(np.linalg.norm(embed))
    final_norm = float(np.linalg.norm(final))

    # How much of the final representation is the original embedding?
    if embed_norm > 1e-10 and final_norm > 1e-10:
        cos = float(np.dot(embed, final) / (embed_norm * final_norm))
        # Project final onto embedding direction
        proj = float(np.dot(embed, final)) / (embed_norm ** 2)
        embed_retained = proj * embed_norm / final_norm
    else:
        cos = 0.0
        embed_retained = 0.0

    # Layer deltas
    layer_norms = []
    for l in range(model.cfg.n_layers):
        pre = np.array(cache[f'blocks.{l}.hook_resid_pre'][pos])
        post = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        delta = post - pre
        layer_norms.append(float(np.linalg.norm(delta)))

    total_layer_contribution = sum(layer_norms)

    return {
        'embedding_final_cosine': round(cos, 4),
        'embedding_retained_fraction': round(embed_retained, 4),
        'embedding_norm': round(embed_norm, 4),
        'final_norm': round(final_norm, 4),
        'total_layer_contribution': round(total_layer_contribution, 4),
        'per_layer_contribution': [round(n, 4) for n in layer_norms],
    }


def shortcut_detection(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Detect shortcut paths: early layers that strongly predict the final output.

    If the prediction is already determined at an early layer, later
    layers may be superfluous (for this input).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to track (default: top prediction).

    Returns:
        Dict with shortcut detection results.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        intermediate_logits = resid @ W_U + b_U

        top_pred = int(np.argmax(intermediate_logits))
        target_logit = float(intermediate_logits[target_token])
        top_logit = float(np.max(intermediate_logits))

        # Probability of target
        probs = np.exp(intermediate_logits - np.max(intermediate_logits))
        probs = probs / np.sum(probs)
        target_prob = float(probs[target_token])

        per_layer.append({
            'layer': l,
            'top_prediction': top_pred,
            'correct': top_pred == target_token,
            'target_logit': round(target_logit, 4),
            'target_probability': round(target_prob, 4),
        })

    # Find earliest correct layer
    earliest_correct = None
    for p in per_layer:
        if p['correct']:
            earliest_correct = p['layer']
            break

    return {
        'target_token': target_token,
        'per_layer': per_layer,
        'earliest_correct_layer': earliest_correct,
        'has_shortcut': earliest_correct is not None and earliest_correct < model.cfg.n_layers - 1,
    }


def minimal_circuit_depth(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    logit_threshold: float = 0.9,
) -> dict:
    """Estimate the minimum number of layers needed for this prediction.

    Measures what fraction of the final logit is achieved at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        logit_threshold: Fraction of final logit to achieve.

    Returns:
        Dict with minimal depth estimation.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    target_token = int(jnp.argmax(logits[pos]))
    W_U = np.array(model.unembed.W_U[:, target_token])

    final_resid = np.array(cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'][pos])
    final_logit = float(np.dot(final_resid, W_U))

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        logit_here = float(np.dot(resid, W_U))
        fraction = logit_here / final_logit if abs(final_logit) > 1e-10 else 0.0

        per_layer.append({
            'layer': l,
            'logit': round(logit_here, 4),
            'fraction_of_final': round(fraction, 4),
        })

    # Find minimal depth
    min_depth = model.cfg.n_layers
    for p in per_layer:
        if p['fraction_of_final'] >= logit_threshold:
            min_depth = p['layer'] + 1
            break

    return {
        'target_token': target_token,
        'final_logit': round(final_logit, 4),
        'per_layer': per_layer,
        'minimal_depth': min_depth,
        'total_layers': model.cfg.n_layers,
        'threshold': logit_threshold,
    }
