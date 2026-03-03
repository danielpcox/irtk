"""Activation steering utilities.

Tools for modifying model behavior by adding/removing direction vectors
from activations during the forward pass:
- add_vector: Add a steering vector at a specific hook point
- subtract_vector: Remove a direction from activations
- compute_steering_vector: Extract a steering vector from contrasting prompts
- steer_generation: Apply steering during autoregressive generation
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def add_vector(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    vector: jnp.ndarray,
    alpha: float = 1.0,
    pos: Optional[int] = None,
) -> jnp.ndarray:
    """Run the model with a steering vector added at a hook point.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point name where the vector is added
            (e.g., "blocks.5.hook_resid_post").
        vector: [d_model] direction vector to add.
        alpha: Scaling factor for the vector.
        pos: If given, only add the vector at this position.
            Otherwise, add to all positions.

    Returns:
        [seq_len, d_vocab] logits with the steering vector applied.
    """
    def steer_hook(x, name):
        scaled = vector * alpha
        if pos is not None:
            return x.at[pos].add(scaled)
        else:
            return x + scaled[None, :] if scaled.ndim == 1 else x + scaled

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steer_hook)])


def subtract_vector(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    vector: jnp.ndarray,
    alpha: float = 1.0,
    pos: Optional[int] = None,
) -> jnp.ndarray:
    """Run the model with a direction removed from activations at a hook point.

    Projects out the component of activations along the given direction.
    This is different from add_vector with negative alpha: it removes
    the projection rather than adding a negative vector.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point name to intervene on.
        vector: [d_model] direction to remove.
        alpha: How much of the projection to remove (1.0 = full removal).
        pos: If given, only modify this position.

    Returns:
        [seq_len, d_vocab] logits with the direction removed.
    """
    # Normalize the direction
    direction = vector / (jnp.linalg.norm(vector) + 1e-10)

    def remove_hook(x, name):
        if pos is not None:
            proj = jnp.dot(x[pos], direction) * direction
            return x.at[pos].add(-alpha * proj)
        else:
            # Project out for all positions: x - alpha * (x . d) * d
            projections = x @ direction  # [seq_len]
            return x - alpha * projections[:, None] * direction[None, :]

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, remove_hook)])


def compute_steering_vector(
    model: HookedTransformer,
    positive_tokens: list[jnp.ndarray],
    negative_tokens: list[jnp.ndarray],
    hook_name: str,
    pos: int = -1,
) -> np.ndarray:
    """Compute a steering vector as the mean activation difference.

    Runs both sets of prompts through the model, collects activations
    at the specified hook point, and returns the mean difference:
        vector = mean(positive_activations) - mean(negative_activations)

    Args:
        model: HookedTransformer.
        positive_tokens: List of token sequences representing the positive direction.
        negative_tokens: List of token sequences representing the negative direction.
        hook_name: Hook point to collect activations from.
        pos: Position to take activations from (-1 for last).

    Returns:
        [d_model] steering vector.
    """
    def _collect_activations(token_sequences):
        all_acts = []
        for tokens in token_sequences:
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                act = cache.cache_dict[hook_name][pos]  # [d_model]
                all_acts.append(np.array(act))
        if not all_acts:
            return np.zeros(model.cfg.d_model)
        return np.mean(all_acts, axis=0)

    pos_mean = _collect_activations(positive_tokens)
    neg_mean = _collect_activations(negative_tokens)

    return pos_mean - neg_mean


def steer_generation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    vector: jnp.ndarray,
    alpha: float = 1.0,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    pos: Optional[int] = None,
) -> jnp.ndarray:
    """Generate tokens autoregressively with a steering vector applied.

    At each step, runs the model with the steering vector added at the
    specified hook point, samples the next token, and appends it.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] initial tokens (prompt).
        hook_name: Hook point to add the steering vector.
        vector: [d_model] steering vector.
        alpha: Scaling factor.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        pos: If given, only add the vector at this position.

    Returns:
        [seq_len + max_new_tokens] array including original and generated tokens.
    """
    current_tokens = jnp.array(tokens)
    key = jax.random.PRNGKey(0)

    for i in range(max_new_tokens):
        # Truncate to context length
        if len(current_tokens) > model.cfg.n_ctx:
            current_tokens = current_tokens[-model.cfg.n_ctx:]

        logits = add_vector(model, current_tokens, hook_name, vector, alpha, pos)
        next_logits = logits[-1]  # [d_vocab]

        if temperature <= 0:
            next_token = jnp.argmax(next_logits)
        else:
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits / temperature)

        current_tokens = jnp.concatenate([current_tokens, next_token[None]])

    return current_tokens


def activation_diff_at_hook(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
) -> np.ndarray:
    """Compute the activation difference between two inputs at a hook point.

    Useful for computing steering vectors from single contrasting pairs.

    Args:
        model: HookedTransformer.
        tokens_a: [seq_len] first input tokens.
        tokens_b: [seq_len] second input tokens.
        hook_name: Hook point to compare activations at.
        pos: Position to take activations from (-1 for last).

    Returns:
        [d_model] difference vector (a - b).
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)

    act_a = cache_a.cache_dict.get(hook_name)
    act_b = cache_b.cache_dict.get(hook_name)

    if act_a is None or act_b is None:
        return np.zeros(model.cfg.d_model)

    return np.array(act_a[pos] - act_b[pos])
