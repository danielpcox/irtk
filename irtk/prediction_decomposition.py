"""Prediction decomposition: how predictions form step by step.

Full decomposition of the prediction pipeline:
- Embedding contribution to logits
- Per-layer logit buildup
- Attention vs MLP logit share
- Prediction confidence evolution
- Alternative prediction tracking
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def logit_buildup(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Track how the target logit builds up through layers.

    Shows the cumulative logit contribution from embedding through
    each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to track (default: top prediction).

    Returns:
        Dict with cumulative logit at each layer.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U[:, target_token])

    steps = []

    # Embedding contribution
    embed = np.array(cache['blocks.0.hook_resid_pre'][pos])
    embed_logit = float(np.dot(embed, W_U))
    steps.append({
        'stage': 'embedding',
        'cumulative_logit': round(embed_logit, 4),
        'delta': round(embed_logit, 4),
    })

    cumulative = embed_logit
    for l in range(model.cfg.n_layers):
        attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
        mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])

        attn_logit = float(np.dot(attn, W_U))
        mlp_logit = float(np.dot(mlp, W_U))
        layer_delta = attn_logit + mlp_logit
        cumulative += layer_delta

        steps.append({
            'stage': f'layer_{l}',
            'cumulative_logit': round(cumulative, 4),
            'delta': round(layer_delta, 4),
            'attn_delta': round(attn_logit, 4),
            'mlp_delta': round(mlp_logit, 4),
        })

    return {
        'target_token': target_token,
        'steps': steps,
        'final_logit': round(cumulative, 4),
    }


def attn_vs_mlp_logit_share(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Compare total attention vs MLP contribution to top-k logits.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of tokens to analyze.

    Returns:
        Dict with attn/MLP share for each top token.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    top_tokens = np.array(jnp.argsort(logits[pos])[::-1][:top_k])
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    per_token = []
    for tok in top_tokens:
        target_dir = W_U[:, int(tok)]

        total_attn = 0.0
        total_mlp = 0.0
        for l in range(model.cfg.n_layers):
            attn = np.array(cache[f'blocks.{l}.hook_attn_out'][pos])
            mlp = np.array(cache[f'blocks.{l}.hook_mlp_out'][pos])
            total_attn += float(np.dot(attn, target_dir))
            total_mlp += float(np.dot(mlp, target_dir))

        embed = np.array(cache['blocks.0.hook_resid_pre'][pos])
        embed_logit = float(np.dot(embed, target_dir))

        total = abs(total_attn) + abs(total_mlp) + abs(embed_logit)
        per_token.append({
            'token': int(tok),
            'logit': round(float(logits[pos, int(tok)]), 4),
            'embed_contribution': round(embed_logit, 4),
            'attn_contribution': round(total_attn, 4),
            'mlp_contribution': round(total_mlp, 4),
            'attn_share': round(abs(total_attn) / total, 4) if total > 0 else 0.0,
            'mlp_share': round(abs(total_mlp) / total, 4) if total > 0 else 0.0,
        })

    return {
        'per_token': per_token,
        'pos': pos,
    }


def confidence_evolution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Track how prediction confidence changes through layers.

    Uses the residual at each layer to compute intermediate logits
    and measure confidence (max probability).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-layer confidence and top prediction.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    per_layer = []

    # After embedding
    resid = np.array(cache['blocks.0.hook_resid_pre'][pos])
    logits_0 = resid @ W_U + b_U
    probs_0 = np.exp(logits_0 - np.max(logits_0))
    probs_0 = probs_0 / np.sum(probs_0)
    per_layer.append({
        'stage': 'embedding',
        'top_token': int(np.argmax(probs_0)),
        'confidence': round(float(np.max(probs_0)), 4),
        'entropy': round(float(-np.sum(probs_0 * np.log(probs_0 + 1e-10))), 4),
    })

    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        logits_l = resid @ W_U + b_U
        probs_l = np.exp(logits_l - np.max(logits_l))
        probs_l = probs_l / np.sum(probs_l)

        per_layer.append({
            'stage': f'layer_{l}',
            'top_token': int(np.argmax(probs_l)),
            'confidence': round(float(np.max(probs_l)), 4),
            'entropy': round(float(-np.sum(probs_l * np.log(probs_l + 1e-10))), 4),
        })

    # Detect when prediction commits (stabilizes)
    final_token = per_layer[-1]['top_token']
    commit_layer = None
    for i in range(len(per_layer) - 1, -1, -1):
        if per_layer[i]['top_token'] != final_token:
            commit_layer = i + 1 if i + 1 < len(per_layer) else len(per_layer) - 1
            break
    if commit_layer is None:
        commit_layer = 0

    return {
        'per_layer': per_layer,
        'final_token': final_token,
        'commit_stage': per_layer[commit_layer]['stage'],
    }


def alternative_predictions(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Track how alternative (runner-up) predictions evolve through layers.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of alternatives to track.

    Returns:
        Dict with per-layer rankings of top-k tokens.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    # Final top-k
    final_top = np.array(jnp.argsort(logits[pos])[::-1][:top_k])

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    per_layer = []

    for l in range(model.cfg.n_layers):
        resid = np.array(cache[f'blocks.{l}.hook_resid_post'][pos])
        logits_l = resid @ W_U + b_U

        # Rank of each final top-k token at this layer
        rankings = []
        sorted_idx = np.argsort(logits_l)[::-1]
        for tok in final_top:
            rank = int(np.where(sorted_idx == tok)[0][0])
            rankings.append({
                'token': int(tok),
                'logit': round(float(logits_l[int(tok)]), 4),
                'rank': rank,
            })

        per_layer.append({
            'layer': l,
            'rankings': rankings,
        })

    return {
        'final_top_tokens': [int(t) for t in final_top],
        'per_layer': per_layer,
    }


def embedding_logit_bias(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Analyze how the embedding alone biases the prediction.

    The embedding sets the initial logit distribution before any
    layer processing. This function shows what the model would predict
    from the embedding alone.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of top predictions to show.

    Returns:
        Dict with embedding-only predictions and bias analysis.
    """
    _, cache = model.run_with_cache(tokens)

    resid = np.array(cache['blocks.0.hook_resid_pre'][pos])
    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    logits = resid @ W_U + b_U
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)

    top_tokens = np.argsort(logits)[::-1][:top_k]

    embed_predictions = []
    for tok in top_tokens:
        embed_predictions.append({
            'token': int(tok),
            'logit': round(float(logits[int(tok)]), 4),
            'probability': round(float(probs[int(tok)]), 4),
        })

    # Compare with final prediction
    final_logits = np.array(model(tokens)[pos])
    final_top = int(np.argmax(final_logits))
    embed_rank_of_final = int(np.where(np.argsort(logits)[::-1] == final_top)[0][0])

    return {
        'embed_predictions': embed_predictions,
        'embed_top_token': int(top_tokens[0]),
        'final_top_token': final_top,
        'embed_rank_of_final': embed_rank_of_final,
        'embed_entropy': round(float(-np.sum(probs * np.log(probs + 1e-10))), 4),
    }
