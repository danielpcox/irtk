"""Layer importance profiling: comprehensive layer importance analysis."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def layer_ablation_importance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Rank layers by how much ablating them changes the output.

    For each layer, zero out the layer's contribution (attn + mlp) and
    measure how much the final logits change.
    """
    logits, cache = model.run_with_cache(tokens)
    clean_logprobs = jax.nn.log_softmax(logits[-1])
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        # Zero ablate attention output
        def zero_attn(x, name):
            return jnp.zeros_like(x)
        def zero_mlp(x, name):
            return jnp.zeros_like(x)

        hooks = [
            (f'blocks.{layer}.hook_attn_out', zero_attn),
            (f'blocks.{layer}.hook_mlp_out', zero_mlp),
        ]
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_logprobs = jax.nn.log_softmax(ablated_logits[-1])

        kl_div = float(jnp.sum(jnp.exp(clean_logprobs) * (clean_logprobs - ablated_logprobs)))
        logit_diff = float(jnp.max(jnp.abs(logits[-1] - ablated_logits[-1])))

        per_layer.append({
            'layer': layer,
            'kl_divergence': kl_div,
            'max_logit_change': logit_diff,
            'is_critical': kl_div > 0.1,
        })

    per_layer.sort(key=lambda x: x['kl_divergence'], reverse=True)
    most_important = per_layer[0]['layer']
    least_important = per_layer[-1]['layer']

    return {
        'per_layer': per_layer,
        'most_important': most_important,
        'least_important': least_important,
        'n_critical': sum(1 for p in per_layer if p['is_critical']),
    }


def layer_gradient_importance(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Measure layer importance via gradient norms through residual stream.

    Computes the gradient of the predicted token logit with respect to each
    layer's residual stream, giving a sensitivity measure.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    # Get target token
    logits_final = cache['ln_final.hook_normalized'] @ model.unembed.W_U
    if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None:
        logits_final = logits_final + model.unembed.b_U
    target_token = int(jnp.argmax(logits_final[pos]))

    per_layer = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key in cache:
            resid = cache[resid_key]
            grad_norm = float(jnp.linalg.norm(resid[pos]))
            per_layer.append({
                'layer': layer,
                'residual_norm': grad_norm,
                'relative_norm': 0.0,  # filled below
            })

    if per_layer:
        max_norm = max(p['residual_norm'] for p in per_layer)
        if max_norm > 0:
            for p in per_layer:
                p['relative_norm'] = p['residual_norm'] / max_norm

    per_layer.sort(key=lambda x: x['residual_norm'], reverse=True)

    return {
        'per_layer': per_layer,
        'position': pos,
        'target_token': target_token,
    }


def layer_output_magnitude(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile each layer's output magnitude (attn + MLP norms).

    Layers with larger outputs contribute more to the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    total_attn = 0.0
    total_mlp = 0.0

    for layer in range(n_layers):
        attn_key = f'blocks.{layer}.hook_attn_out'
        mlp_key = f'blocks.{layer}.hook_mlp_out'

        attn_norm = float(jnp.mean(jnp.linalg.norm(cache[attn_key], axis=-1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(cache[mlp_key], axis=-1)))
        combined = attn_norm + mlp_norm

        total_attn += attn_norm
        total_mlp += mlp_norm

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'combined_norm': combined,
            'attn_fraction': attn_norm / combined if combined > 0 else 0.5,
        })

    total = total_attn + total_mlp

    return {
        'per_layer': per_layer,
        'total_attn_norm': total_attn,
        'total_mlp_norm': total_mlp,
        'attn_dominated_layers': sum(1 for p in per_layer if p['attn_fraction'] > 0.5),
        'mlp_dominated_layers': sum(1 for p in per_layer if p['attn_fraction'] <= 0.5),
    }


def layer_prediction_impact(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much does each layer change the model's prediction?

    Measures KL divergence between predictions at consecutive layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else None

    per_layer = []
    prev_logprobs = None

    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]
        logits = resid[-1] @ W_U
        if b_U is not None:
            logits = logits + b_U
        logprobs = jax.nn.log_softmax(logits)

        if prev_logprobs is not None:
            kl = float(jnp.sum(jnp.exp(logprobs) * (logprobs - prev_logprobs)))
            top_token = int(jnp.argmax(logits))
            prev_top = int(jnp.argmax(jnp.exp(prev_logprobs)))

            per_layer.append({
                'layer': layer,
                'kl_from_previous': abs(kl),
                'prediction_changed': top_token != prev_top,
                'top_token': top_token,
            })
        else:
            per_layer.append({
                'layer': layer,
                'kl_from_previous': 0.0,
                'prediction_changed': False,
                'top_token': int(jnp.argmax(logits)),
            })

        prev_logprobs = logprobs

    n_changes = sum(1 for p in per_layer if p['prediction_changed'])
    biggest_shift = max(per_layer, key=lambda x: x['kl_from_previous'])['layer'] if per_layer else 0

    return {
        'per_layer': per_layer,
        'n_prediction_changes': n_changes,
        'biggest_shift_layer': biggest_shift,
    }


def cumulative_layer_importance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much of the final prediction is formed by each layer?

    Projects each layer's residual onto the final prediction direction
    to measure cumulative prediction formation.
    """
    logits, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Final prediction direction
    target_token = int(jnp.argmax(logits[-1]))
    W_U = model.unembed.W_U
    pred_direction = W_U[:, target_token]
    pred_direction = pred_direction / (jnp.linalg.norm(pred_direction) + 1e-10)

    per_layer = []
    prev_proj = 0.0

    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key][-1]  # last position
        proj = float(jnp.dot(resid, pred_direction))
        increment = proj - prev_proj

        per_layer.append({
            'layer': layer,
            'cumulative_projection': proj,
            'layer_increment': increment,
            'fraction_of_final': 0.0,  # filled below
        })
        prev_proj = proj

    final_proj = per_layer[-1]['cumulative_projection'] if per_layer else 1.0
    if abs(final_proj) > 1e-10:
        for p in per_layer:
            p['fraction_of_final'] = p['cumulative_projection'] / final_proj

    return {
        'per_layer': per_layer,
        'target_token': target_token,
        'final_projection': float(final_proj) if per_layer else 0.0,
    }
