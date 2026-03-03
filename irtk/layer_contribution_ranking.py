"""Layer contribution ranking: rank layers by their impact on the final output."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def layer_logit_contribution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Rank layers by their contribution to the target token logit.

    Projects each layer's output through unembedding.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Final prediction
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post'][pos]
    final_logits = final_resid @ W_U
    target_token = int(jnp.argmax(final_logits))

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]

        attn_logit = float((attn_out @ W_U)[target_token])
        mlp_logit = float((mlp_out @ W_U)[target_token])
        total_logit = attn_logit + mlp_logit

        per_layer.append({
            'layer': layer,
            'attn_logit_contrib': attn_logit,
            'mlp_logit_contrib': mlp_logit,
            'total_logit_contrib': total_logit,
        })

    per_layer.sort(key=lambda x: abs(x['total_logit_contrib']), reverse=True)

    return {
        'position': pos,
        'target_token': target_token,
        'per_layer': per_layer,
    }


def layer_norm_contribution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Rank layers by the magnitude of their output.

    Larger output = more modification of the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_norm = float(jnp.linalg.norm(cache[f'blocks.{layer}.hook_attn_out'][pos]))
        mlp_norm = float(jnp.linalg.norm(cache[f'blocks.{layer}.hook_mlp_out'][pos]))
        total_norm = (attn_norm ** 2 + mlp_norm ** 2) ** 0.5

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'total_norm': total_norm,
        })

    per_layer.sort(key=lambda x: x['total_norm'], reverse=True)

    return {
        'position': pos,
        'per_layer': per_layer,
    }


def layer_direction_importance(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Rank layers by alignment with the final residual direction.

    High alignment = layer output is in the same direction as the final result.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post'][pos]
    final_dir = final_resid / (jnp.linalg.norm(final_resid) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]
        combined = attn_out + mlp_out

        projection = float(jnp.dot(combined, final_dir))
        cosine = float(jnp.dot(combined, final_dir) / (jnp.linalg.norm(combined) + 1e-10))

        per_layer.append({
            'layer': layer,
            'projection': projection,
            'cosine_with_final': cosine,
            'is_constructive': bool(projection > 0),
        })

    per_layer.sort(key=lambda x: x['projection'], reverse=True)

    return {
        'position': pos,
        'per_layer': per_layer,
        'n_constructive': sum(1 for p in per_layer if p['is_constructive']),
    }


def layer_entropy_contribution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How does each layer change prediction entropy?

    Entropy decrease = sharpening the prediction.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(n_layers):
        pre_resid = cache[f'blocks.{layer}.hook_resid_pre'][pos]
        post_resid = cache[f'blocks.{layer}.hook_resid_post'][pos]

        pre_logits = pre_resid @ W_U
        post_logits = post_resid @ W_U

        pre_probs = jax.nn.softmax(pre_logits)
        post_probs = jax.nn.softmax(post_logits)

        pre_entropy = -float(jnp.sum(pre_probs * jnp.log(pre_probs + 1e-10)))
        post_entropy = -float(jnp.sum(post_probs * jnp.log(post_probs + 1e-10)))

        per_layer.append({
            'layer': layer,
            'pre_entropy': pre_entropy,
            'post_entropy': post_entropy,
            'entropy_change': post_entropy - pre_entropy,
            'sharpens': bool(post_entropy < pre_entropy),
        })

    return {
        'position': pos,
        'per_layer': per_layer,
        'n_sharpening': sum(1 for p in per_layer if p['sharpens']),
    }


def layer_cumulative_effect(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Cumulative effect of layers on the target token logit.

    Shows how the prediction builds up through layers.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post'][pos]
    final_logits = final_resid @ W_U
    target_token = int(jnp.argmax(final_logits))

    cumulative = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
        logits = resid @ W_U
        target_logit = float(logits[target_token])
        rank = int(jnp.sum(logits > logits[target_token]))

        cumulative.append({
            'layer': layer,
            'target_logit': target_logit,
            'target_rank': rank,
            'is_top1': bool(rank == 0),
        })

    return {
        'position': pos,
        'target_token': target_token,
        'cumulative': cumulative,
    }
