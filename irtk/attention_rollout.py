"""Attention rollout and flow-based attribution.

Tools for computing effective attention across multiple layers:
- attention_rollout: Multiply attention matrices across layers
- attention_flow: Account for residual stream in attention computation
- effective_attention: Per-layer attention accounting for residual mixing
- layer_aggregated_attention: Aggregate attention across all heads in a layer
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def layer_aggregated_attention(
    cache: ActivationCache,
    layer: int,
    n_heads: int,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate attention patterns across heads for a single layer.

    Args:
        cache: ActivationCache from run_with_cache.
        layer: Layer index.
        n_heads: Number of heads.
        method: Aggregation method ("mean", "max", or "min").

    Returns:
        [seq_len, seq_len] aggregated attention matrix.
    """
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    if hook_name not in cache.cache_dict:
        raise KeyError(f"Attention pattern not found: {hook_name}")

    # pattern shape: [seq_q, n_heads, seq_k]
    pattern = np.array(cache.cache_dict[hook_name])

    if method == "mean":
        return np.mean(pattern, axis=1)  # [seq_q, seq_k]
    elif method == "max":
        return np.max(pattern, axis=1)
    elif method == "min":
        return np.min(pattern, axis=1)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: 'mean', 'max', 'min'")


def effective_attention(
    cache: ActivationCache,
    layer: int,
    n_heads: int,
    residual_weight: float = 0.5,
    method: str = "mean",
) -> np.ndarray:
    """Compute effective attention for a layer including residual stream.

    The residual stream acts as an identity connection, so the effective
    attention matrix is a mixture of the attention pattern and the identity:

        A_eff = (1 - w) * I + w * A

    where w is the residual_weight (how much of the signal comes from attention
    vs the residual stream).

    Args:
        cache: ActivationCache.
        layer: Layer index.
        n_heads: Number of heads.
        residual_weight: Weight given to attention (vs identity). Range [0, 1].
        method: Head aggregation method.

    Returns:
        [seq_len, seq_len] effective attention matrix.
    """
    attn = layer_aggregated_attention(cache, layer, n_heads, method=method)
    seq_len = attn.shape[0]
    identity = np.eye(seq_len)

    return (1 - residual_weight) * identity + residual_weight * attn


def attention_rollout(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    head_aggregation: str = "mean",
) -> np.ndarray:
    """Compute attention rollout by multiplying attention matrices across layers.

    Attention rollout (Abnar & Zuidema 2020) tracks how attention flows
    from input to output by chaining attention matrices:

        R = A_L @ A_{L-1} @ ... @ A_1

    where each A_l is the head-aggregated attention pattern for layer l.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        start_layer: First layer to include.
        end_layer: Last layer to include (exclusive). Default: all layers.
        head_aggregation: How to combine heads ("mean", "max").

    Returns:
        [seq_len, seq_len] rollout matrix. Entry [i, j] is the effective
        attention from output position i to input position j.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    if end_layer is None:
        end_layer = model.cfg.n_layers

    seq_len = len(tokens)
    rollout = np.eye(seq_len)

    for layer in range(start_layer, end_layer):
        attn = layer_aggregated_attention(
            cache, layer, model.cfg.n_heads, method=head_aggregation
        )
        rollout = attn @ rollout

    return rollout


def attention_flow(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    residual_weight: float = 0.5,
    head_aggregation: str = "mean",
) -> np.ndarray:
    """Compute attention flow with residual stream contribution.

    Like attention rollout but accounts for the residual stream by
    mixing each layer's attention with the identity:

        R = A_eff_L @ A_eff_{L-1} @ ... @ A_eff_1

    where A_eff_l = (1-w)*I + w*A_l.

    This better reflects that the residual stream carries information
    unchanged through layers.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        start_layer: First layer.
        end_layer: Last layer (exclusive).
        residual_weight: Weight for attention vs identity per layer.
        head_aggregation: How to combine heads.

    Returns:
        [seq_len, seq_len] flow matrix.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    if end_layer is None:
        end_layer = model.cfg.n_layers

    seq_len = len(tokens)
    flow = np.eye(seq_len)

    for layer in range(start_layer, end_layer):
        attn_eff = effective_attention(
            cache, layer, model.cfg.n_heads,
            residual_weight=residual_weight,
            method=head_aggregation,
        )
        flow = attn_eff @ flow

    return flow


def token_attribution_rollout(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    output_pos: int = -1,
    method: str = "rollout",
    **kwargs,
) -> np.ndarray:
    """Attribute each input token's contribution to an output position.

    Uses attention rollout or flow to compute how much each input token
    contributes to the output at a specific position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        output_pos: Output position to attribute (-1 for last).
        method: "rollout" for plain rollout, "flow" for residual-aware flow.
        **kwargs: Passed to the underlying rollout/flow function.

    Returns:
        [seq_len] attribution scores per input token (sums to ~1).
    """
    if method == "rollout":
        R = attention_rollout(model, tokens, **kwargs)
    elif method == "flow":
        R = attention_flow(model, tokens, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'rollout' or 'flow'")

    return R[output_pos]


def per_head_rollout(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    output_pos: int = -1,
) -> np.ndarray:
    """Compute attention rollout through a specific head.

    Uses mean aggregation for all other layers but isolates a single
    head at the specified layer to measure that head's contribution
    to the overall attention flow.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer of the head to isolate.
        head: Head index to isolate.
        output_pos: Output position.

    Returns:
        [seq_len] attribution through the specified head.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    rollout = np.eye(seq_len)

    for l in range(n_layers):
        if l == layer:
            # Use only the specified head
            hook_name = f"blocks.{l}.attn.hook_pattern"
            pattern = np.array(cache.cache_dict[hook_name])
            attn = pattern[:, head, :]  # [seq_q, seq_k]
        else:
            attn = layer_aggregated_attention(
                cache, l, model.cfg.n_heads, method="mean"
            )
        rollout = attn @ rollout

    return rollout[output_pos]
