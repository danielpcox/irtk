"""Logit lens surgery.

Surgical logit lens: intervene at specific layers and measure
downstream prediction effects, compare logit lens at different
stages, and track how interventions propagate.
"""

import jax
import jax.numpy as jnp


def logit_lens_at_layer(model, tokens, layer):
    """Apply logit lens at a specific layer.

    Projects intermediate residual stream through the unembedding matrix.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with predictions and confidence at this layer.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
    logits = resid @ model.unembed.W_U + model.unembed.b_U  # [seq, d_vocab]
    probs = jax.nn.softmax(logits, axis=-1)

    seq_len = logits.shape[0]
    per_position = []
    for pos in range(seq_len):
        top_token = int(jnp.argmax(logits[pos]))
        confidence = float(probs[pos, top_token])
        entropy = -float(jnp.sum(probs[pos] * jnp.log(jnp.maximum(probs[pos], 1e-10))))
        per_position.append({
            'position': pos,
            'top_token': top_token,
            'confidence': confidence,
            'entropy': entropy,
        })

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_confidence': float(jnp.mean(jnp.array([p['confidence'] for p in per_position]))),
        'mean_entropy': float(jnp.mean(jnp.array([p['entropy'] for p in per_position]))),
    }


def logit_lens_diff(model, tokens, layer_a, layer_b):
    """Compare logit lens predictions between two layers.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer_a: first layer
        layer_b: second layer

    Returns:
        dict with per-position prediction changes.
    """
    _, cache = model.run_with_cache(tokens)
    resid_a = cache[f'blocks.{layer_a}.hook_resid_post']
    resid_b = cache[f'blocks.{layer_b}.hook_resid_post']

    logits_a = resid_a @ model.unembed.W_U + model.unembed.b_U
    logits_b = resid_b @ model.unembed.W_U + model.unembed.b_U
    probs_a = jax.nn.softmax(logits_a, axis=-1)
    probs_b = jax.nn.softmax(logits_b, axis=-1)

    seq_len = logits_a.shape[0]
    per_position = []
    n_changed = 0
    for pos in range(seq_len):
        top_a = int(jnp.argmax(logits_a[pos]))
        top_b = int(jnp.argmax(logits_b[pos]))
        changed = top_a != top_b
        if changed:
            n_changed += 1

        # KL divergence
        kl = float(jnp.sum(probs_b[pos] * jnp.log(
            jnp.maximum(probs_b[pos], 1e-10) / jnp.maximum(probs_a[pos], 1e-10))))

        per_position.append({
            'position': pos,
            'top_token_a': top_a,
            'top_token_b': top_b,
            'prediction_changed': bool(changed),
            'kl_divergence': kl,
        })

    return {
        'layer_a': layer_a,
        'layer_b': layer_b,
        'per_position': per_position,
        'n_changed': n_changed,
        'change_fraction': n_changed / max(seq_len, 1),
        'mean_kl': float(jnp.mean(jnp.array([p['kl_divergence'] for p in per_position]))),
    }


def logit_lens_intervention(model, tokens, layer, hook_name, hook_fn):
    """Apply logit lens before and after an intervention.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer at which to measure logit lens
        hook_name: hook point for intervention
        hook_fn: intervention function

    Returns:
        dict with prediction changes from intervention.
    """
    # Clean logit lens
    _, clean_cache = model.run_with_cache(tokens)
    clean_resid = clean_cache[f'blocks.{layer}.hook_resid_post']
    clean_logits = clean_resid @ model.unembed.W_U + model.unembed.b_U

    # Intervened: use run_with_hooks with a caching hook
    captured = {}
    resid_hook_name = f'blocks.{layer}.hook_resid_post'

    def capture_resid(x, name):
        captured['resid'] = x
        return x

    model.run_with_hooks(tokens, fwd_hooks=[
        (hook_name, hook_fn),
        (resid_hook_name, capture_resid),
    ])
    mod_resid = captured['resid']
    mod_logits = mod_resid @ model.unembed.W_U + model.unembed.b_U

    diff = mod_logits - clean_logits
    seq_len = diff.shape[0]

    per_position = []
    for pos in range(seq_len):
        clean_top = int(jnp.argmax(clean_logits[pos]))
        mod_top = int(jnp.argmax(mod_logits[pos]))
        per_position.append({
            'position': pos,
            'clean_top': clean_top,
            'modified_top': mod_top,
            'changed': clean_top != mod_top,
            'max_logit_change': float(jnp.max(jnp.abs(diff[pos]))),
        })

    return {
        'layer': layer,
        'hook_name': hook_name,
        'per_position': per_position,
        'n_changed': sum(1 for p in per_position if p['changed']),
        'mean_max_change': float(jnp.mean(jnp.array([p['max_logit_change'] for p in per_position]))),
    }


def component_logit_lens_effect(model, tokens, layer):
    """Compare logit lens before and after each component at a layer.

    Shows how attention and MLP each change the logit lens prediction.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index

    Returns:
        dict with component effects on logit lens.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[f'blocks.{layer}.hook_resid_pre']
    attn_out = cache[f'blocks.{layer}.hook_attn_out']
    mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
    resid_post = cache[f'blocks.{layer}.hook_resid_post']

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    logits_pre = resid_pre @ W_U + b_U
    logits_mid = (resid_pre + attn_out) @ W_U + b_U
    logits_post = resid_post @ W_U + b_U

    seq_len = logits_pre.shape[0]

    per_position = []
    for pos in range(seq_len):
        top_pre = int(jnp.argmax(logits_pre[pos]))
        top_mid = int(jnp.argmax(logits_mid[pos]))
        top_post = int(jnp.argmax(logits_post[pos]))

        attn_change = float(jnp.max(jnp.abs(logits_mid[pos] - logits_pre[pos])))
        mlp_change = float(jnp.max(jnp.abs(logits_post[pos] - logits_mid[pos])))

        per_position.append({
            'position': pos,
            'pre_top': top_pre,
            'post_attn_top': top_mid,
            'post_mlp_top': top_post,
            'attn_changed_prediction': top_pre != top_mid,
            'mlp_changed_prediction': top_mid != top_post,
            'attn_max_logit_change': attn_change,
            'mlp_max_logit_change': mlp_change,
        })

    return {
        'layer': layer,
        'per_position': per_position,
        'attn_changes': sum(1 for p in per_position if p['attn_changed_prediction']),
        'mlp_changes': sum(1 for p in per_position if p['mlp_changed_prediction']),
    }


def logit_lens_trajectory(model, tokens, position=-1, top_k=5):
    """Track how the logit lens prediction evolves across all layers.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: token position to track (-1 for last)
        top_k: number of top tokens to track

    Returns:
        dict with per-layer logit lens predictions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    stages = []
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']
        logits = resid[pos] @ W_U + b_U  # [d_vocab]
        probs = jax.nn.softmax(logits)

        top_indices = jnp.argsort(-probs)[:top_k]
        top_tokens = [{'token': int(t), 'prob': float(probs[t]),
                       'logit': float(logits[t])} for t in top_indices]

        stages.append({
            'layer': l,
            'top_predictions': top_tokens,
            'entropy': -float(jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-10)))),
        })

    return {
        'position': pos,
        'stages': stages,
        'final_prediction': stages[-1]['top_predictions'][0]['token'] if stages else -1,
    }
