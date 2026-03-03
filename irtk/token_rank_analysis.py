"""Token rank analysis: how token rankings evolve through the model.

Tools for tracking where tokens sit in the vocabulary ranking:
- Per-layer token rank tracking
- Rank stability detection
- Rank transition analysis
- Competing token identification
- Rank entropy measurement
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def token_rank_trajectory(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    track_tokens: Optional[list[int]] = None,
) -> dict:
    """Track the rank of specific tokens through each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        track_tokens: Tokens to track (default: top-5 from final layer).

    Returns:
        Dict with per-layer ranks for tracked tokens.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    if track_tokens is None:
        track_tokens = list(np.argsort(np.array(logits[pos]))[::-1][:5])

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        intermediate_logits = resid @ W_U + b_U
        sorted_idx = np.argsort(intermediate_logits)[::-1]

        ranks = {}
        for tok in track_tokens:
            rank = int(np.where(sorted_idx == tok)[0][0])
            ranks[tok] = rank

        per_layer.append({
            'layer': l,
            'ranks': ranks,
            'top_token': int(sorted_idx[0]),
        })

    return {
        'tracked_tokens': track_tokens,
        'per_layer': per_layer,
    }


def rank_stability(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Measure how stable the top-k ranking is across layers.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Size of ranking to compare.

    Returns:
        Dict with ranking stability metrics.
    """
    _, cache = model.run_with_cache(tokens)

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    prev_top = None
    transitions = []

    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        intermediate_logits = resid @ W_U + b_U
        top = set(np.argsort(intermediate_logits)[::-1][:top_k])

        if prev_top is not None:
            overlap = len(top & prev_top)
            transitions.append({
                'from_layer': l - 1,
                'to_layer': l,
                'overlap': overlap,
                'stability': round(overlap / top_k, 4),
            })

        prev_top = top

    mean_stability = float(np.mean([t['stability'] for t in transitions])) if transitions else 1.0

    return {
        'top_k': top_k,
        'transitions': transitions,
        'mean_stability': round(mean_stability, 4),
        'most_unstable': min(transitions, key=lambda x: x['stability']) if transitions else None,
    }


def rank_entropy(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure the entropy of the probability distribution at each layer.

    Low entropy = model is confident. High entropy = spread across tokens.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer entropy values.
    """
    _, cache = model.run_with_cache(tokens)

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        logits_l = resid @ W_U + b_U
        probs = np.exp(logits_l - np.max(logits_l))
        probs = probs / np.sum(probs)

        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(len(probs)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        per_layer.append({
            'layer': l,
            'entropy': round(entropy, 4),
            'normalized_entropy': round(normalized_entropy, 4),
            'max_probability': round(float(np.max(probs)), 4),
        })

    return {
        'per_layer': per_layer,
        'entropy_reduction': round(
            per_layer[0]['entropy'] - per_layer[-1]['entropy'], 4
        ) if len(per_layer) >= 2 else 0.0,
    }


def competing_tokens(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    margin: float = 2.0,
) -> dict:
    """Identify tokens competing for the top prediction at each layer.

    Tokens within `margin` logits of the top prediction are competing.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        margin: Logit margin to count as competing.

    Returns:
        Dict with competing tokens at each layer.
    """
    _, cache = model.run_with_cache(tokens)

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    per_layer = []
    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        logits_l = resid @ W_U + b_U

        top_logit = float(np.max(logits_l))
        competitors = np.where(logits_l > top_logit - margin)[0]

        per_layer.append({
            'layer': l,
            'n_competitors': len(competitors),
            'top_token': int(np.argmax(logits_l)),
            'top_logit': round(top_logit, 4),
            'competitor_tokens': [int(t) for t in competitors[:10]],
        })

    return {
        'per_layer': per_layer,
        'margin': margin,
    }


def rank_change_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Attribute rank changes to attention vs MLP at each layer.

    For a target token, shows which component (attn or MLP) is responsible
    for its rank moving up or down.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to track (default: top prediction).

    Returns:
        Dict with per-layer rank change attribution.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U_target = np.array(model.unembed.W_U[:, target_token])

    per_layer = []
    for l in range(model.cfg.n_layers):
        attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])

        attn_logit = float(np.dot(attn, W_U_target))
        mlp_logit = float(np.dot(mlp, W_U_target))

        per_layer.append({
            'layer': l,
            'attn_logit_change': round(attn_logit, 4),
            'mlp_logit_change': round(mlp_logit, 4),
            'total_change': round(attn_logit + mlp_logit, 4),
            'main_driver': 'attn' if abs(attn_logit) > abs(mlp_logit) else 'mlp',
        })

    return {
        'target_token': target_token,
        'per_layer': per_layer,
    }
