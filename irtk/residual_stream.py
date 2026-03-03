"""Residual stream analysis utilities.

Tools for analyzing the residual stream across layers:
- cosine_similarity_to_unembed: How aligned is each layer with the output direction
- residual_norm_by_layer: Track residual stream norm growth
- residual_direction_analysis: Project components onto a specific direction
- token_prediction_trajectory: Track how predictions evolve across layers
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def cosine_similarity_to_unembed(
    model: HookedTransformer,
    cache: ActivationCache,
    token: int,
    pos: int = -1,
) -> np.ndarray:
    """Compute cosine similarity of each layer's residual with a token's unembed direction.

    At each layer, measures how aligned the residual stream is with the
    direction in embedding space that promotes a specific token.

    Args:
        model: HookedTransformer.
        cache: ActivationCache from run_with_cache.
        token: Token ID to measure alignment with.
        pos: Sequence position (-1 for last).

    Returns:
        [n_layers+1] array of cosine similarities (embed + each layer).
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    target_dir = W_U[:, token]  # [d_model]
    target_norm = float(jnp.linalg.norm(target_dir))

    resid_stack = cache.accumulated_resid()  # [n_components, seq, d_model]
    n_components = resid_stack.shape[0]

    similarities = np.zeros(n_components)
    for i in range(n_components):
        resid = resid_stack[i, pos]
        resid_norm = float(jnp.linalg.norm(resid))
        if resid_norm > 1e-10 and target_norm > 1e-10:
            similarities[i] = float(jnp.dot(resid, target_dir)) / (resid_norm * target_norm)

    return similarities


def residual_norm_by_layer(
    cache: ActivationCache,
    pos: Optional[int] = None,
) -> np.ndarray:
    """Track the L2 norm of the residual stream at each layer.

    The residual stream typically grows in norm across layers.
    This helps diagnose training issues or understand model behavior.

    Args:
        cache: ActivationCache from run_with_cache.
        pos: If specified, only compute for this position. Otherwise, mean across positions.

    Returns:
        [n_layers+1] array of residual stream norms.
    """
    resid_stack = cache.accumulated_resid()  # [n_components, seq, d_model]
    n_components = resid_stack.shape[0]

    norms = np.zeros(n_components)
    for i in range(n_components):
        if pos is not None:
            norms[i] = float(jnp.linalg.norm(resid_stack[i, pos]))
        else:
            # Mean norm across positions
            per_pos = jnp.linalg.norm(resid_stack[i], axis=-1)  # [seq]
            norms[i] = float(jnp.mean(per_pos))

    return norms


def residual_direction_analysis(
    model: HookedTransformer,
    cache: ActivationCache,
    direction: jnp.ndarray,
    pos: int = -1,
) -> dict[str, np.ndarray]:
    """Decompose each component's contribution along a specific direction.

    For a given direction in d_model space, shows how much each component
    (embedding, each attention, each MLP) contributes along that direction.

    Args:
        model: HookedTransformer.
        cache: ActivationCache from run_with_cache.
        direction: [d_model] unit direction to project onto.
        pos: Sequence position (-1 for last).

    Returns:
        Dict with:
        - "components": [n_components] array of projections
        - "labels": list of component names
        - "cumulative": [n_components] running total
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    components = []
    labels = []

    # Embedding
    embed = cache.cache_dict.get("hook_embed", None)
    pos_embed = cache.cache_dict.get("hook_pos_embed", None)
    if embed is not None:
        e = embed[pos]
        if pos_embed is not None:
            e = e + pos_embed[pos]
        components.append(float(jnp.dot(e, direction)))
        labels.append("embed")

    # Each layer
    for layer in range(model.cfg.n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        if attn_key in cache.cache_dict:
            proj = float(jnp.dot(cache.cache_dict[attn_key][pos], direction))
            components.append(proj)
            labels.append(f"L{layer}_attn")

        if mlp_key in cache.cache_dict:
            proj = float(jnp.dot(cache.cache_dict[mlp_key][pos], direction))
            components.append(proj)
            labels.append(f"L{layer}_mlp")

    components_arr = np.array(components)
    cumulative = np.cumsum(components_arr)

    return {
        "components": components_arr,
        "labels": labels,
        "cumulative": cumulative,
    }


def token_prediction_trajectory(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    k: int = 5,
) -> list[list[tuple[int, float]]]:
    """Track how the model's top-k predictions change across layers.

    At each layer, projects the residual stream through ln_final + W_U
    and reports the top-k predicted tokens with their probabilities.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to track (-1 for last).
        k: Number of top predictions.

    Returns:
        List of [top-k predictions at each layer]. Each prediction is (token_id, probability).
    """
    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()  # [n_components, seq, d_model]

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    trajectory = []
    for i in range(resid_stack.shape[0]):
        resid = resid_stack[i, pos]  # [d_model]

        # Apply final LN if present
        if model.ln_final is not None:
            # LN on single position: reshape to [1, d_model]
            resid_2d = resid[None, :]
            normed = model.ln_final(resid_2d, None)
            resid = normed[0]

        logits = resid @ W_U + b_U  # [d_vocab]
        probs = jax.nn.softmax(logits)

        top_indices = jnp.argsort(probs)[::-1][:k]
        layer_preds = [(int(idx), float(probs[idx])) for idx in top_indices]
        trajectory.append(layer_preds)

    return trajectory


def prediction_rank_trajectory(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> np.ndarray:
    """Track where a target token ranks in the prediction at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token to track the rank of.
        pos: Position to track.

    Returns:
        [n_layers+1] array of ranks (0 = top prediction).
    """
    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    ranks = np.zeros(resid_stack.shape[0])
    for i in range(resid_stack.shape[0]):
        resid = resid_stack[i, pos]

        if model.ln_final is not None:
            resid_2d = resid[None, :]
            normed = model.ln_final(resid_2d, None)
            resid = normed[0]

        logits = resid @ W_U + b_U
        sorted_indices = jnp.argsort(logits)[::-1]
        rank = int(jnp.where(sorted_indices == target_token, size=1)[0][0])
        ranks[i] = rank

    return ranks


def layer_contribution_to_logit(
    model: HookedTransformer,
    cache: ActivationCache,
    token: int,
    pos: int = -1,
) -> dict[str, float]:
    """Compute how much each layer's output contributes to a logit.

    This is a thin wrapper around residual_direction_analysis using
    the unembed direction for a specific token.

    Args:
        model: HookedTransformer.
        cache: ActivationCache.
        token: Target token ID.
        pos: Position.

    Returns:
        Dict mapping component name -> logit contribution.
    """
    W_U = model.unembed.W_U
    direction = W_U[:, token]
    result = residual_direction_analysis(model, cache, direction, pos=pos)
    return dict(zip(result["labels"], result["components"]))
