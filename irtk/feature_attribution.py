"""Feature-level attribution tools.

Tools for understanding which input features drive internal representations
and model predictions:
- token_to_neuron_attribution: Which input tokens drive a specific neuron
- token_to_direction_attribution: Which tokens contribute to a specific direction
- decompose_logit_by_token: Per-input-token contribution to a logit
- feature_importance_ranking: Rank most important features at a hook point
- cross_layer_attribution: How activations at one layer contribute to another
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.hook_points import HookState


def _run_with_hooks_and_cache(model, tokens, fwd_hooks):
    """Run model with both hooks and caching enabled."""
    cache_dict = {}
    hook_fns = {name: fn for name, fn in fwd_hooks}
    hook_state = HookState(hook_fns=hook_fns, cache=cache_dict)
    logits = model(tokens, hook_state)
    from irtk.activation_cache import ActivationCache
    return logits, ActivationCache(cache_dict, model)


def token_to_neuron_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    neuron: int,
) -> np.ndarray:
    """Attribute a neuron's activation to input tokens via leave-one-out ablation.

    For each input token, zeros its embedding and measures how much
    the target neuron's activation changes.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: MLP layer index.
        neuron: Neuron index within the MLP.

    Returns:
        [seq_len] attribution score per input token.
    """
    tokens = jnp.array(tokens)
    hook_name = f"blocks.{layer}.mlp.hook_post"

    # Baseline
    _, cache = model.run_with_cache(tokens)
    if hook_name not in cache.cache_dict:
        return np.zeros(len(tokens))

    baseline = float(cache.cache_dict[hook_name][-1, neuron])

    # Per-token ablation
    attributions = []
    for pos in range(len(tokens)):
        def ablate_hook(x, name, p=pos):
            return x.at[p].set(jnp.zeros(x.shape[-1]))

        _, abl_cache = _run_with_hooks_and_cache(
            model, tokens, [("hook_embed", ablate_hook)]
        )
        if hook_name in abl_cache.cache_dict:
            ablated = float(abl_cache.cache_dict[hook_name][-1, neuron])
        else:
            ablated = 0.0
        attributions.append(abs(baseline - ablated))

    return np.array(attributions)


def token_to_direction_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    direction: np.ndarray,
    pos: int = -1,
) -> np.ndarray:
    """Attribute a direction's activation to input tokens.

    For each input token, measures how much ablating that token's embedding
    changes the projection of the activation onto the direction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point to measure the direction at.
        direction: [d_model] direction vector.
        pos: Position to measure at (-1 for last).

    Returns:
        [seq_len] attribution score per input token.
    """
    tokens = jnp.array(tokens)
    direction = np.array(direction, dtype=np.float32)
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction = direction / d_norm

    # Baseline
    _, cache = model.run_with_cache(tokens)
    if hook_name not in cache.cache_dict:
        return np.zeros(len(tokens))

    baseline_act = np.array(cache.cache_dict[hook_name][pos])
    baseline_proj = float(np.dot(baseline_act, direction))

    # Per-token ablation
    attributions = []
    for token_pos in range(len(tokens)):
        def ablate_hook(x, name, tp=token_pos):
            return x.at[tp].set(jnp.zeros(x.shape[-1]))

        _, abl_cache = _run_with_hooks_and_cache(
            model, tokens, [("hook_embed", ablate_hook)]
        )
        if hook_name in abl_cache.cache_dict:
            abl_act = np.array(abl_cache.cache_dict[hook_name][pos])
            abl_proj = float(np.dot(abl_act, direction))
        else:
            abl_proj = 0.0
        attributions.append(abs(baseline_proj - abl_proj))

    return np.array(attributions)


def decompose_logit_by_token(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict:
    """Decompose a logit prediction into per-input-token contributions.

    For each input token position, measures how much it contributes to
    the target token's logit via leave-one-out ablation.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token ID whose logit to decompose.
        pos: Output position to analyze (-1 for last).

    Returns:
        Dict with:
        - "attributions": [seq_len] per-token contribution to target logit
        - "baseline_logit": The clean target logit value
        - "total_attribution": Sum of all token attributions
    """
    tokens = jnp.array(tokens)

    # Baseline logit
    logits = model(tokens)
    baseline = float(logits[pos, target_token])

    # Per-token ablation
    attributions = []
    for token_pos in range(len(tokens)):
        def ablate_hook(x, name, tp=token_pos):
            return x.at[tp].set(jnp.zeros(x.shape[-1]))

        abl_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[("hook_embed", ablate_hook)],
        )
        abl_val = float(abl_logits[pos, target_token])
        attributions.append(baseline - abl_val)

    attributions = np.array(attributions)

    return {
        "attributions": attributions,
        "baseline_logit": baseline,
        "total_attribution": float(np.sum(attributions)),
    }


def feature_importance_ranking(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    k: int = 10,
    pos: int = -1,
) -> dict:
    """Rank the most important activation dimensions at a hook point.

    Importance is measured by the magnitude of each dimension of the
    activation vector.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point to analyze.
        k: Number of top features to return.
        pos: Position to analyze (-1 for last).

    Returns:
        Dict with:
        - "top_indices": [k] indices of the most important features
        - "top_values": [k] activation values of the top features
        - "top_magnitudes": [k] absolute values of the top features
        - "full_activations": [d] full activation vector
    """
    tokens = jnp.array(tokens)

    _, cache = model.run_with_cache(tokens)
    if hook_name not in cache.cache_dict:
        return {
            "top_indices": np.array([], dtype=int),
            "top_values": np.array([]),
            "top_magnitudes": np.array([]),
            "full_activations": np.array([]),
        }

    act = np.array(cache.cache_dict[hook_name][pos])  # [d]
    magnitudes = np.abs(act)

    k = min(k, len(act))
    top_idx = np.argsort(magnitudes)[::-1][:k]

    return {
        "top_indices": top_idx,
        "top_values": act[top_idx],
        "top_magnitudes": magnitudes[top_idx],
        "full_activations": act,
    }


def cross_layer_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    source_hook: str,
    target_hook: str,
    target_direction: np.ndarray,
    pos: int = -1,
    n_dims: int = 10,
) -> dict:
    """Attribute a target direction's activation to source layer dimensions.

    For each of the top-n dimensions at the source layer, measures how
    much zeroing that dimension changes the target direction's projection.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        source_hook: Earlier layer hook to attribute from.
        target_hook: Later layer hook where the direction is measured.
        target_direction: [d_model] direction to measure at target.
        pos: Position to analyze (-1 for last).
        n_dims: Number of source dimensions to analyze.

    Returns:
        Dict with:
        - "source_dims": [n_dims] indices of source dimensions analyzed
        - "attributions": [n_dims] how much each source dim contributes
        - "baseline_projection": Clean projection value onto target direction
    """
    tokens = jnp.array(tokens)
    direction = np.array(target_direction, dtype=np.float32)
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-10:
        direction = direction / d_norm

    # Baseline
    _, cache = model.run_with_cache(tokens)
    if target_hook not in cache.cache_dict or source_hook not in cache.cache_dict:
        return {
            "source_dims": np.array([], dtype=int),
            "attributions": np.array([]),
            "baseline_projection": 0.0,
        }

    target_act = np.array(cache.cache_dict[target_hook][pos])
    baseline_proj = float(np.dot(target_act, direction))

    source_act = np.array(cache.cache_dict[source_hook][pos])
    source_mags = np.abs(source_act)

    # Top-n source dimensions
    n_dims = min(n_dims, len(source_act))
    top_source_dims = np.argsort(source_mags)[::-1][:n_dims]

    # For each source dimension, zero it and measure target change
    attributions = []
    for dim in top_source_dims:
        def zero_dim_hook(x, name, d=dim):
            return x.at[pos, d].set(0.0)

        _, abl_cache = _run_with_hooks_and_cache(
            model, tokens, [(source_hook, zero_dim_hook)]
        )
        if target_hook in abl_cache.cache_dict:
            abl_target = np.array(abl_cache.cache_dict[target_hook][pos])
            abl_proj = float(np.dot(abl_target, direction))
        else:
            abl_proj = 0.0
        attributions.append(baseline_proj - abl_proj)

    return {
        "source_dims": top_source_dims,
        "attributions": np.array(attributions),
        "baseline_projection": baseline_proj,
    }
