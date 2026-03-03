"""Attention pattern surgery tools.

Surgical manipulation of attention patterns for mechanistic analysis:
- attention_knockout: Zero specific (query, key) attention entries
- attention_knockout_matrix: Knockout with a boolean mask
- attention_pattern_patch: Patch attention pattern from another run
- force_attention: Force a head to attend to a specific pattern
- attention_edge_attribution: Per-edge attribution matrix
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def attention_knockout(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    query_pos: int,
    key_pos: int,
) -> jnp.ndarray:
    """Knock out a specific (query, key) attention edge.

    Sets the pre-softmax attention score to -inf for the specified
    (query, key) pair, then lets softmax renormalize.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        layer: Attention layer index.
        head: Attention head index.
        query_pos: Query position (negative indexing supported).
        key_pos: Key position (negative indexing supported).

    Returns:
        Logits from the model with the attention edge knocked out.
    """
    hook_name = f"blocks.{layer}.attn.hook_attn_scores"

    def knockout_hook(x, name):
        # x shape: [seq_q, n_heads, seq_k]
        return x.at[query_pos, head, key_pos].set(jnp.finfo(x.dtype).min)

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, knockout_hook)])


def attention_knockout_matrix(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    mask: np.ndarray,
) -> jnp.ndarray:
    """Knock out attention edges specified by a boolean mask.

    For each True entry in mask[q, k], sets the pre-softmax attention
    score to -inf.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        layer: Attention layer index.
        head: Attention head index.
        mask: [seq_q, seq_k] boolean mask. True = knock out.

    Returns:
        Logits from the model with the masked edges knocked out.
    """
    hook_name = f"blocks.{layer}.attn.hook_attn_scores"
    mask_arr = jnp.array(mask, dtype=bool)

    def knockout_hook(x, name):
        # x: [seq_q, n_heads, seq_k]
        # mask_arr: [seq_q, seq_k]
        neg_inf = jnp.finfo(x.dtype).min
        # Expand mask to match: [seq_q, 1, seq_k] and apply only to target head
        head_scores = x[:, head, :]  # [seq_q, seq_k]
        head_scores = jnp.where(mask_arr, neg_inf, head_scores)
        return x.at[:, head, :].set(head_scores)

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, knockout_hook)])


def attention_pattern_patch(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    layer: int,
    head: int,
) -> jnp.ndarray:
    """Patch attention pattern from corrupted run into clean run.

    Replaces the post-softmax attention pattern of a specific head
    while keeping everything else from the clean run.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean input tokens.
        corrupted_tokens: Corrupted input tokens.
        layer: Attention layer index.
        head: Attention head index.

    Returns:
        Logits from the clean run with the attention pattern patched.
    """
    # Get corrupted attention pattern
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    _, cache = model.run_with_cache(corrupted_tokens)

    if hook_name not in cache.cache_dict:
        return model(clean_tokens)

    corrupted_pattern = cache.cache_dict[hook_name]  # [seq_q, n_heads, seq_k]

    def patch_hook(x, name):
        return x.at[:, head, :].set(corrupted_pattern[:, head, :])

    return model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, patch_hook)])


def force_attention(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    target_pattern: np.ndarray,
) -> jnp.ndarray:
    """Force a head to use a specific attention pattern.

    Replaces the post-softmax attention pattern for the specified head
    with the given target pattern.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        layer: Attention layer index.
        head: Attention head index.
        target_pattern: [seq_q, seq_k] attention pattern (should sum to 1 per row).

    Returns:
        Logits with the forced attention pattern.
    """
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    pattern = jnp.array(target_pattern)

    def force_hook(x, name):
        return x.at[:, head, :].set(pattern)

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, force_hook)])


def attention_edge_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    metric_fn: Callable[[jnp.ndarray], float],
) -> np.ndarray:
    """Compute per-edge attribution by knocking out each (q, k) pair.

    For each (query, key) position, measures the metric change when
    that attention edge is removed.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        layer: Attention layer index.
        head: Attention head index.
        metric_fn: Function(logits) -> float.

    Returns:
        [seq_q, seq_k] attribution matrix. Larger values = more important edges.
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    # Baseline
    baseline_logits = model(tokens)
    baseline_metric = metric_fn(baseline_logits)

    # Per-edge knockout
    attribution = np.zeros((seq_len, seq_len))
    for q in range(seq_len):
        for k in range(q + 1):  # causal: can only attend to k <= q
            logits = attention_knockout(model, tokens, layer, head, q, k)
            ablated_metric = metric_fn(logits)
            attribution[q, k] = abs(baseline_metric - ablated_metric)

    return attribution
