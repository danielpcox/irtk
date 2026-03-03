"""Text generation utilities for interpretability.

Provides autoregressive generation with caching and sampling strategies:
- generate: Basic text generation with configurable sampling
- generate_with_cache: Generate while caching activations at each step
- generate_comparison: Generate from two models side by side
- top_k_sampling, nucleus_sampling: Sampling strategy functions
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def top_k_sampling(logits: jnp.ndarray, k: int = 50) -> jnp.ndarray:
    """Apply top-k filtering to logits.

    Sets all logits outside the top-k to -inf.

    Args:
        logits: [d_vocab] raw logits.
        k: Number of top logits to keep.

    Returns:
        [d_vocab] filtered logits.
    """
    top_k_vals = jnp.sort(logits)[::-1][:k]
    threshold = top_k_vals[-1]
    return jnp.where(logits >= threshold, logits, -jnp.inf)


def nucleus_sampling(logits: jnp.ndarray, p: float = 0.9) -> jnp.ndarray:
    """Apply nucleus (top-p) filtering to logits.

    Keeps the smallest set of tokens whose cumulative probability exceeds p.

    Args:
        logits: [d_vocab] raw logits.
        p: Cumulative probability threshold.

    Returns:
        [d_vocab] filtered logits.
    """
    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probs = jax.nn.softmax(sorted_logits)
    cumsum = jnp.cumsum(probs)

    # Find cutoff: keep tokens until cumsum exceeds p
    # Include the first token that pushes cumsum over p
    mask = cumsum - probs < p
    filtered = jnp.where(mask, sorted_logits, -jnp.inf)

    # Unsort back to original order
    result = jnp.empty_like(logits).at[sorted_indices].set(filtered)
    return result


def generate(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_token: Optional[int] = None,
    seed: int = 0,
) -> jnp.ndarray:
    """Generate tokens autoregressively.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] prompt tokens.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0 = greedy.
        top_k: If set, apply top-k filtering.
        top_p: If set, apply nucleus sampling.
        stop_token: If set, stop generation when this token is produced.
        seed: Random seed for sampling.

    Returns:
        [seq_len + n_generated] array of all tokens (prompt + generated).
    """
    current = jnp.array(tokens)
    key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        # Truncate to context window
        input_tokens = current
        if len(input_tokens) > model.cfg.n_ctx:
            input_tokens = input_tokens[-model.cfg.n_ctx:]

        logits = model(input_tokens)
        next_logits = logits[-1]  # [d_vocab]

        # Apply filtering
        if top_k is not None:
            next_logits = top_k_sampling(next_logits, k=top_k)
        if top_p is not None:
            next_logits = nucleus_sampling(next_logits, p=top_p)

        # Sample
        if temperature <= 0:
            next_token = jnp.argmax(next_logits)
        else:
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits / temperature)

        current = jnp.concatenate([current, next_token[None]])

        if stop_token is not None and int(next_token) == stop_token:
            break

    return current


def generate_with_cache(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    hook_names: Optional[list[str]] = None,
    seed: int = 0,
) -> tuple[jnp.ndarray, list[ActivationCache]]:
    """Generate tokens while caching activations at each step.

    This is useful for analyzing the model's internal state during
    generation (e.g., tracking attention patterns or residual stream
    evolution step by step).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] prompt tokens.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. 0 = greedy.
        hook_names: If given, only cache these specific hook points.
            Otherwise, cache everything.
        seed: Random seed.

    Returns:
        (generated_tokens, caches) where:
        - generated_tokens: [seq_len + n_generated] full sequence
        - caches: list of ActivationCache, one per generation step
    """
    current = jnp.array(tokens)
    key = jax.random.PRNGKey(seed)
    all_caches = []

    for _ in range(max_new_tokens):
        input_tokens = current
        if len(input_tokens) > model.cfg.n_ctx:
            input_tokens = input_tokens[-model.cfg.n_ctx:]

        logits, cache = model.run_with_cache(input_tokens)

        # Optionally filter to specific hook names
        if hook_names is not None:
            filtered = {k: v for k, v in cache.cache_dict.items() if k in hook_names}
            cache = ActivationCache(filtered, model)

        all_caches.append(cache)

        next_logits = logits[-1]

        if temperature <= 0:
            next_token = jnp.argmax(next_logits)
        else:
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits / temperature)

        current = jnp.concatenate([current, next_token[None]])

    return current, all_caches


def generate_comparison(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Generate from two models on the same prompt and compare outputs.

    Both models generate independently from the same prompt.
    This is useful for comparing finetuned vs. base models.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: [seq_len] prompt tokens.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        seed: Random seed (same for both models).

    Returns:
        Dict with:
        - "tokens_a": generated sequence from model_a
        - "tokens_b": generated sequence from model_b
        - "diverge_pos": position where outputs first differ (-1 if identical)
    """
    tokens_a = generate(model_a, tokens, max_new_tokens, temperature, seed=seed)
    tokens_b = generate(model_b, tokens, max_new_tokens, temperature, seed=seed)

    # Find first divergence point
    min_len = min(len(tokens_a), len(tokens_b))
    diverge_pos = -1
    for i in range(min_len):
        if int(tokens_a[i]) != int(tokens_b[i]):
            diverge_pos = i
            break

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "diverge_pos": diverge_pos,
    }
