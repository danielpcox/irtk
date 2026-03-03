"""Attention pattern analysis utilities.

Provides tools for analyzing attention patterns from cached activations:
- entropy: Per-head attention entropy (diffuse vs focused)
- head_pattern_similarity: Cosine similarity between attention patterns
- attention_to_token: How much attention flows to a given token
- causal_tracing: Corrupt a token, measure recovery at each layer
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def entropy(
    cache: ActivationCache,
    layer: int,
    head: int,
) -> np.ndarray:
    """Compute attention entropy for each query position.

    High entropy means the head distributes attention broadly.
    Low entropy means focused/sharp attention (e.g., attending to one token).

    Args:
        cache: ActivationCache from run_with_cache.
        layer: Layer index.
        head: Head index.

    Returns:
        [seq_len] array of entropy values per query position.
    """
    pattern = np.array(cache[("pattern", layer)][head])  # [q, k]
    # Clip for numerical stability
    pattern = np.clip(pattern, 1e-10, 1.0)
    ent = -np.sum(pattern * np.log(pattern), axis=-1)  # [q]
    return ent


def all_head_entropy(cache: ActivationCache, model: HookedTransformer) -> np.ndarray:
    """Compute mean attention entropy for every head.

    Args:
        cache: ActivationCache.
        model: HookedTransformer.

    Returns:
        [n_layers, n_heads] array of mean entropy per head.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    result = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            result[l, h] = np.mean(entropy(cache, l, h))
    return result


def head_pattern_similarity(
    cache: ActivationCache,
    layer1: int, head1: int,
    layer2: int, head2: int,
) -> float:
    """Compute cosine similarity between two heads' attention patterns.

    Flattens each head's attention pattern into a vector and computes
    cosine similarity. Useful for finding heads that attend to similar things.

    Args:
        cache: ActivationCache.
        layer1, head1: First head.
        layer2, head2: Second head.

    Returns:
        Cosine similarity in [-1, 1].
    """
    p1 = np.array(cache[("pattern", layer1)][head1]).flatten()
    p2 = np.array(cache[("pattern", layer2)][head2]).flatten()

    norm1 = np.linalg.norm(p1)
    norm2 = np.linalg.norm(p2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(p1, p2) / (norm1 * norm2))


def attention_to_token(
    cache: ActivationCache,
    layer: int,
    head: int,
    key_pos: int,
) -> np.ndarray:
    """Get how much attention each query position pays to a specific key position.

    Args:
        cache: ActivationCache.
        layer: Layer index.
        head: Head index.
        key_pos: The key/source position to measure attention to.

    Returns:
        [seq_len] array of attention weights to key_pos from each query.
    """
    pattern = np.array(cache[("pattern", layer)][head])  # [q, k]
    return pattern[:, key_pos]


def attention_from_token(
    cache: ActivationCache,
    layer: int,
    head: int,
    query_pos: int,
) -> np.ndarray:
    """Get the attention distribution from a specific query position.

    Args:
        cache: ActivationCache.
        layer: Layer index.
        head: Head index.
        query_pos: The query position.

    Returns:
        [seq_len] array of attention weights from query_pos to all keys.
    """
    pattern = np.array(cache[("pattern", layer)][head])  # [q, k]
    return pattern[query_pos]


def top_attended_tokens(
    cache: ActivationCache,
    layer: int,
    head: int,
    query_pos: int = -1,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find the top-k key positions attended to by a query position.

    Args:
        cache: ActivationCache.
        layer: Layer index.
        head: Head index.
        query_pos: Query position (-1 for last).
        k: Number of top positions to return.

    Returns:
        List of (position, attention_weight) tuples, sorted descending.
    """
    attn = attention_from_token(cache, layer, head, query_pos)
    top_indices = np.argsort(attn)[::-1][:k]
    return [(int(idx), float(attn[idx])) for idx in top_indices]


def causal_tracing(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    corrupt_pos: int,
    metric_fn: Callable[[jnp.ndarray], float],
    noise_std: float = 0.1,
) -> dict[str, np.ndarray]:
    """Perform causal tracing: corrupt a token and measure recovery.

    Corrupts the embedding at a specific position with noise, then for each
    layer, patches in the clean residual stream at that layer and measures
    how much the metric recovers.

    This reveals which layers are most important for processing the
    information at the corrupted position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        corrupt_pos: Position to corrupt.
        metric_fn: Function(logits) -> float measuring the behavior of interest.
        noise_std: Standard deviation of Gaussian noise for corruption.

    Returns:
        Dict with:
        - "clean": clean metric value
        - "corrupted": corrupted metric value
        - "restored_resid": [n_layers] array of metric values when restoring
          the residual stream at each layer
        - "restored_attn": [n_layers] array when restoring attention output
        - "restored_mlp": [n_layers] array when restoring MLP output
    """
    n_layers = model.cfg.n_layers

    # Clean run
    clean_logits = model(tokens)
    clean_metric = metric_fn(clean_logits)

    # Get clean cache
    _, clean_cache = model.run_with_cache(tokens)

    # Corrupted run: add noise to embedding at corrupt_pos
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (model.cfg.d_model,)) * noise_std

    def corrupt_embed(x, name):
        return x.at[corrupt_pos].add(noise)

    corrupted_logits = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", corrupt_embed)]
    )
    corrupted_metric = metric_fn(corrupted_logits)

    # For each layer, restore the clean activation and measure recovery
    restored_resid = np.zeros(n_layers)
    restored_attn = np.zeros(n_layers)
    restored_mlp = np.zeros(n_layers)

    for layer in range(n_layers):
        # Restore residual stream (at corrupt_pos only)
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key in clean_cache.cache_dict:
            clean_resid = clean_cache.cache_dict[resid_key]

            def restore_resid(x, name, _cr=clean_resid, _p=corrupt_pos):
                return x.at[_p].set(_cr[_p])

            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", corrupt_embed), (resid_key, restore_resid)],
            )
            restored_resid[layer] = metric_fn(logits)

        # Restore attention output
        attn_key = f"blocks.{layer}.hook_attn_out"
        if attn_key in clean_cache.cache_dict:
            clean_attn = clean_cache.cache_dict[attn_key]

            def restore_attn(x, name, _ca=clean_attn, _p=corrupt_pos):
                return x.at[_p].set(_ca[_p])

            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", corrupt_embed), (attn_key, restore_attn)],
            )
            restored_attn[layer] = metric_fn(logits)

        # Restore MLP output
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if mlp_key in clean_cache.cache_dict:
            clean_mlp = clean_cache.cache_dict[mlp_key]

            def restore_mlp(x, name, _cm=clean_mlp, _p=corrupt_pos):
                return x.at[_p].set(_cm[_p])

            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", corrupt_embed), (mlp_key, restore_mlp)],
            )
            restored_mlp[layer] = metric_fn(logits)

    return {
        "clean": clean_metric,
        "corrupted": corrupted_metric,
        "restored_resid": restored_resid,
        "restored_attn": restored_attn,
        "restored_mlp": restored_mlp,
    }


def max_attention_position(
    cache: ActivationCache,
    layer: int,
    head: int,
) -> np.ndarray:
    """Find which key position each query attends to most.

    Args:
        cache: ActivationCache.
        layer: Layer index.
        head: Head index.

    Returns:
        [seq_len] array of key positions (argmax of attention per query).
    """
    pattern = np.array(cache[("pattern", layer)][head])
    return np.argmax(pattern, axis=-1)


def attention_head_summary(
    cache: ActivationCache,
    model: HookedTransformer,
) -> dict[str, np.ndarray]:
    """Compute summary statistics for all attention heads.

    Args:
        cache: ActivationCache.
        model: HookedTransformer.

    Returns:
        Dict with keys:
        - "entropy": [n_layers, n_heads] mean entropy per head
        - "max_attn": [n_layers, n_heads] mean of max attention weight per head
        - "diag_score": [n_layers, n_heads] mean self-attention (diagonal)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    ent = np.zeros((n_layers, n_heads))
    max_attn = np.zeros((n_layers, n_heads))
    diag_score = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            pattern = np.array(cache[("pattern", l)][h])  # [q, k]
            seq_len = pattern.shape[0]

            # Mean entropy
            ent[l, h] = np.mean(entropy(cache, l, h))

            # Mean of max attention
            max_attn[l, h] = np.mean(np.max(pattern, axis=-1))

            # Diagonal score (self-attention)
            diag = np.array([pattern[i, i] for i in range(seq_len)])
            diag_score[l, h] = np.mean(diag)

    return {
        "entropy": ent,
        "max_attn": max_attn,
        "diag_score": diag_score,
    }
