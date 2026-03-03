"""Neuron-level analysis utilities for MLP interpretability.

Tools for understanding what individual MLP neurons compute:
- neuron_activation_stats: Per-neuron firing statistics
- top_activating_tokens: Tokens that maximally activate a neuron
- neuron_to_logit: Project a neuron through W_out @ W_U
- dead_neuron_fraction: Find dead neurons
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def get_neuron_activations(
    cache: ActivationCache,
    layer: int,
    hook: str = "post",
) -> np.ndarray:
    """Get neuron activations from a cache.

    Args:
        cache: ActivationCache from run_with_cache.
        layer: Layer index.
        hook: "pre" (before activation fn) or "post" (after activation fn).

    Returns:
        [seq_len, d_mlp] array of neuron activations.
    """
    key = f"blocks.{layer}.mlp.hook_{hook}"
    return np.array(cache.cache_dict[key])


def neuron_activation_stats(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    layer: int,
    hook: str = "post",
) -> dict[str, np.ndarray]:
    """Compute per-neuron activation statistics across a dataset.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays to analyze.
        layer: Layer index.
        hook: "pre" or "post" activations.

    Returns:
        Dict with:
        - "mean": [d_mlp] mean activation per neuron
        - "std": [d_mlp] std deviation per neuron
        - "max": [d_mlp] max activation per neuron
        - "firing_rate": [d_mlp] fraction of positions where neuron > 0
        - "l1_norm": [d_mlp] mean absolute activation
    """
    all_acts = []
    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        acts = get_neuron_activations(cache, layer, hook)
        all_acts.append(acts)

    # Stack: [total_tokens, d_mlp]
    stacked = np.concatenate(all_acts, axis=0)

    return {
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "max": np.max(stacked, axis=0),
        "firing_rate": np.mean(stacked > 0, axis=0),
        "l1_norm": np.mean(np.abs(stacked), axis=0),
    }


def top_activating_tokens(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    layer: int,
    neuron: int,
    k: int = 10,
    hook: str = "post",
) -> list[tuple[int, int, float]]:
    """Find tokens that maximally activate a specific neuron.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        layer: Layer index.
        neuron: Neuron index within d_mlp.
        k: Number of top activations to return.
        hook: "pre" or "post".

    Returns:
        List of (prompt_idx, position, activation) tuples, sorted descending.
    """
    entries = []
    for prompt_idx, tokens in enumerate(token_sequences):
        _, cache = model.run_with_cache(tokens)
        acts = get_neuron_activations(cache, layer, hook)
        for pos in range(acts.shape[0]):
            entries.append((prompt_idx, pos, float(acts[pos, neuron])))

    entries.sort(key=lambda x: x[2], reverse=True)
    return entries[:k]


def neuron_to_logit(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    k: int = 10,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Project a neuron through W_out @ W_U to see its effect on logits.

    When neuron i fires, it writes W_out[i, :] to the residual stream.
    Projecting through W_U tells us which tokens it promotes/suppresses.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        neuron: Neuron index.
        k: Number of top promoted/suppressed tokens.

    Returns:
        (promoted, suppressed) where each is [(token_id, logit_effect), ...].
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Neuron's writing direction
    neuron_dir = W_out[neuron]  # [d_model]

    # Project through unembedding
    logit_effects = np.array(neuron_dir @ W_U)  # [d_vocab]

    # Top promoted
    top_promoted_idx = np.argsort(logit_effects)[::-1][:k]
    promoted = [(int(idx), float(logit_effects[idx])) for idx in top_promoted_idx]

    # Top suppressed
    top_suppressed_idx = np.argsort(logit_effects)[:k]
    suppressed = [(int(idx), float(logit_effects[idx])) for idx in top_suppressed_idx]

    return promoted, suppressed


def neuron_logit_effects(
    model: HookedTransformer,
    layer: int,
) -> np.ndarray:
    """Compute the logit effect matrix for all neurons in a layer.

    Returns the full W_out @ W_U matrix.

    Args:
        model: HookedTransformer.
        layer: Layer index.

    Returns:
        [d_mlp, d_vocab] array where entry [i, j] is neuron i's effect on token j's logit.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    return np.array(W_out @ W_U)


def dead_neuron_fraction(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    layer: int,
    threshold: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Compute the fraction of neurons that never activate above a threshold.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        layer: Layer index.
        threshold: Activation threshold (default 0 for ReLU-like).

    Returns:
        (fraction_dead, is_dead) where:
        - fraction_dead: scalar fraction of dead neurons
        - is_dead: [d_mlp] boolean array
    """
    ever_active = None

    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        acts = get_neuron_activations(cache, layer, "post")
        active = np.any(acts > threshold, axis=0)
        if ever_active is None:
            ever_active = active
        else:
            ever_active = ever_active | active

    if ever_active is None:
        return 0.0, np.array([])

    is_dead = ~ever_active
    return float(np.mean(is_dead)), is_dead


def neuron_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    layer: int,
    token: int,
    pos: int = -1,
) -> np.ndarray:
    """Compute each neuron's contribution to a specific output logit.

    Attribution = neuron_activation * (W_out[neuron] @ W_U[:, token])

    Args:
        model: HookedTransformer.
        cache: ActivationCache from run_with_cache.
        layer: Layer index.
        token: Target token ID.
        pos: Position to analyze.

    Returns:
        [d_mlp] array of per-neuron attribution values.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Each neuron's logit contribution direction
    logit_dir = W_U[:, token]  # [d_model]
    neuron_logit = np.array(W_out @ logit_dir)  # [d_mlp]

    # Neuron activations at this position
    acts = get_neuron_activations(cache, layer, "post")
    acts_at_pos = acts[pos]  # [d_mlp]

    return acts_at_pos * neuron_logit
