"""Prediction pathway analysis: trace how predictions form through the model."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def prediction_buildup(model: HookedTransformer, tokens: jnp.ndarray,
                        position: int = -1) -> dict:
    """How does the prediction for the next token build up layer by layer?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    if position < 0:
        position = seq_len + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    next_token = int(tokens[min(position + 1, seq_len - 1)])

    embed = cache['hook_embed'][position] + cache['hook_pos_embed'][position]

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        logits = resid @ W_U  # [d_vocab]
        probs = jax.nn.softmax(logits)
        top_token = int(jnp.argmax(logits))

        per_layer.append({
            'layer': layer,
            'top_prediction': top_token,
            'top_logit': float(logits[top_token]),
            'target_logit': float(logits[next_token]),
            'target_prob': float(probs[next_token]),
            'target_rank': int(jnp.sum(logits > logits[next_token])),
        })

    return {
        'position': position,
        'target_token': next_token,
        'per_layer': per_layer,
    }


def prediction_component_attribution(model: HookedTransformer, tokens: jnp.ndarray,
                                      position: int = -1) -> dict:
    """Attribute the final prediction to attention vs MLP at each layer."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    if position < 0:
        position = seq_len + position

    W_U = model.unembed.W_U
    next_token = int(tokens[min(position + 1, seq_len - 1)])
    target_dir = W_U[:, next_token]
    target_dir = target_dir / (jnp.linalg.norm(target_dir) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][position]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][position]

        attn_logit = float(attn_out @ W_U[:, next_token])
        mlp_logit = float(mlp_out @ W_U[:, next_token])

        per_layer.append({
            'layer': layer,
            'attn_logit_contribution': attn_logit,
            'mlp_logit_contribution': mlp_logit,
            'total_contribution': attn_logit + mlp_logit,
            'attn_alignment': float(jnp.dot(attn_out / (jnp.linalg.norm(attn_out) + 1e-10), target_dir)),
            'mlp_alignment': float(jnp.dot(mlp_out / (jnp.linalg.norm(mlp_out) + 1e-10), target_dir)),
        })

    return {
        'position': position,
        'target_token': next_token,
        'per_layer': per_layer,
    }


def prediction_confidence_evolution(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Track prediction confidence (max prob) through layers."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        logits = resid @ W_U
        probs = jax.nn.softmax(logits)
        entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-10)))

        per_layer.append({
            'layer': layer,
            'max_prob': float(jnp.max(probs)),
            'entropy': entropy,
            'top_token': int(jnp.argmax(logits)),
        })

    return {
        'position': position,
        'per_layer': per_layer,
    }


def prediction_competition(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1, top_k: int = 5) -> dict:
    """Which tokens are competing for the prediction at each layer?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        logits = resid @ W_U
        probs = jax.nn.softmax(logits)

        top_indices = jnp.argsort(probs)[-top_k:][::-1]
        candidates = []
        for idx in top_indices:
            idx = int(idx)
            candidates.append({
                'token_id': idx,
                'probability': float(probs[idx]),
                'logit': float(logits[idx]),
            })

        per_layer.append({
            'layer': layer,
            'candidates': candidates,
            'margin': float(probs[top_indices[0]] - probs[top_indices[1]]) if top_k >= 2 else 0,
        })

    return {
        'position': position,
        'per_layer': per_layer,
    }


def prediction_commit_point(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """At which layer does the model commit to its final prediction?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    W_U = model.unembed.W_U

    # Final prediction
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post'][position]
    final_logits = final_resid @ W_U
    final_token = int(jnp.argmax(final_logits))

    per_layer = []
    commit_layer = None
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        logits = resid @ W_U
        top_token = int(jnp.argmax(logits))
        matches_final = top_token == final_token

        per_layer.append({
            'layer': layer,
            'top_token': top_token,
            'matches_final': bool(matches_final),
        })

        if matches_final and commit_layer is None:
            # Check if it stays committed
            stays = True
            for future in range(layer + 1, n_layers):
                future_resid = cache[f'blocks.{future}.hook_resid_post'][position]
                future_top = int(jnp.argmax(future_resid @ W_U))
                if future_top != final_token:
                    stays = False
                    break
            if stays:
                commit_layer = layer

    return {
        'position': position,
        'final_prediction': final_token,
        'commit_layer': commit_layer,
        'per_layer': per_layer,
    }
