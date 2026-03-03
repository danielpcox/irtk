"""Layer specialization profiling.

Profile what each layer specializes in: how much it changes predictions,
what information it adds, and how it compares to other layers.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def layer_prediction_impact(model, tokens, pos=-1):
    """How much does each layer change the prediction?

    Returns:
        dict with per_layer prediction change metrics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    results = []
    for l in range(n_layers):
        pre_key = f'blocks.{l}.hook_resid_pre' if l == 0 else f'blocks.{l-1}.hook_resid_post'
        # Actually use hook_resid_pre for this layer and hook_resid_post
        if l == 0:
            pre_key = 'blocks.0.hook_resid_pre'
        else:
            pre_key = f'blocks.{l-1}.hook_resid_post'
        post_key = f'blocks.{l}.hook_resid_post'

        if pre_key not in cache or post_key not in cache:
            continue

        pre_logits = cache[pre_key][pos] @ W_U + b_U
        post_logits = cache[post_key][pos] @ W_U + b_U

        pre_top = int(jnp.argmax(pre_logits))
        post_top = int(jnp.argmax(post_logits))

        # KL divergence from pre to post
        pre_probs = jax.nn.softmax(pre_logits)
        post_probs = jax.nn.softmax(post_logits)
        kl = float(jnp.sum(post_probs * jnp.log(jnp.maximum(post_probs, 1e-10) / jnp.maximum(pre_probs, 1e-10))))

        results.append({
            'layer': l,
            'prediction_changed': pre_top != post_top,
            'pre_top': pre_top,
            'post_top': post_top,
            'kl_divergence': kl,
            'logit_delta_norm': float(jnp.linalg.norm(post_logits - pre_logits)),
        })

    return {'per_layer': results}


def layer_information_added(model, tokens, pos=-1):
    """What information does each layer add to the residual?

    Returns:
        dict with per_layer information addition metrics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    results = []
    for l in range(n_layers):
        attn_key = f'blocks.{l}.hook_attn_out'
        mlp_key = f'blocks.{l}.hook_mlp_out'
        pre_key = f'blocks.{l}.hook_resid_pre' if l == 0 else f'blocks.{l-1}.hook_resid_post'

        if pre_key not in cache:
            continue

        pre = cache[pre_key][pos]
        pre_norm = float(jnp.linalg.norm(pre))

        attn_norm = 0.0
        mlp_norm = 0.0
        attn_cos = 0.0
        mlp_cos = 0.0

        if attn_key in cache:
            attn = cache[attn_key][pos]
            attn_norm = float(jnp.linalg.norm(attn))
            attn_cos = float(jnp.dot(attn, pre) / jnp.maximum(attn_norm * pre_norm, 1e-10))

        if mlp_key in cache:
            mlp = cache[mlp_key][pos]
            mlp_norm = float(jnp.linalg.norm(mlp))
            mlp_cos = float(jnp.dot(mlp, pre) / jnp.maximum(mlp_norm * pre_norm, 1e-10))

        results.append({
            'layer': l,
            'attn_new_info': attn_norm * (1 - abs(attn_cos)),  # orthogonal component
            'mlp_new_info': mlp_norm * (1 - abs(mlp_cos)),
            'attn_reinforcement': attn_norm * abs(attn_cos),  # parallel component
            'mlp_reinforcement': mlp_norm * abs(mlp_cos),
            'attn_alignment': attn_cos,
            'mlp_alignment': mlp_cos,
        })

    return {'per_layer': results}


def layer_uniqueness(model, tokens, pos=-1):
    """How unique is each layer's contribution compared to others?

    Returns:
        dict with per_layer uniqueness scores.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    # Collect per-layer deltas
    deltas = []
    for l in range(n_layers):
        if l == 0:
            pre_key = 'blocks.0.hook_resid_pre'
        else:
            pre_key = f'blocks.{l-1}.hook_resid_post'
        post_key = f'blocks.{l}.hook_resid_post'

        if pre_key in cache and post_key in cache:
            delta = cache[post_key][pos] - cache[pre_key][pos]
            deltas.append(delta)
        else:
            deltas.append(jnp.zeros(model.cfg.d_model))

    # Compute pairwise cosine similarity
    results = []
    for i in range(len(deltas)):
        similarities = []
        ni = jnp.linalg.norm(deltas[i])
        for j in range(len(deltas)):
            if i == j:
                continue
            nj = jnp.linalg.norm(deltas[j])
            cos = float(jnp.dot(deltas[i], deltas[j]) / jnp.maximum(ni * nj, 1e-10))
            similarities.append(abs(cos))

        mean_sim = sum(similarities) / max(len(similarities), 1)
        uniqueness = 1.0 - mean_sim  # higher = more unique

        results.append({
            'layer': i,
            'uniqueness': uniqueness,
            'mean_similarity_to_others': mean_sim,
            'delta_norm': float(ni),
        })

    return {'per_layer': results}


def attn_vs_mlp_specialization(model, tokens, pos=-1):
    """Compare attention vs MLP contribution type at each layer.

    Returns:
        dict with per_layer specialization breakdown.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    logits = model(tokens)
    target = int(jnp.argmax(logits[pos]))
    W_U_col = W_U[:, target]

    results = []
    for l in range(n_layers):
        attn_key = f'blocks.{l}.hook_attn_out'
        mlp_key = f'blocks.{l}.hook_mlp_out'

        attn_logit = 0.0
        mlp_logit = 0.0
        attn_norm = 0.0
        mlp_norm = 0.0

        if attn_key in cache:
            a = cache[attn_key][pos]
            attn_logit = float(jnp.dot(a, W_U_col))
            attn_norm = float(jnp.linalg.norm(a))
        if mlp_key in cache:
            m = cache[mlp_key][pos]
            mlp_logit = float(jnp.dot(m, W_U_col))
            mlp_norm = float(jnp.linalg.norm(m))

        total_logit = abs(attn_logit) + abs(mlp_logit)
        results.append({
            'layer': l,
            'attn_logit': attn_logit,
            'mlp_logit': mlp_logit,
            'attn_dominant': abs(attn_logit) > abs(mlp_logit),
            'attn_logit_fraction': abs(attn_logit) / max(total_logit, 1e-10),
        })

    return {'target_token': target, 'per_layer': results}


def layer_role_classification(model, tokens, pos=-1):
    """Classify each layer's role: refining, transforming, or passing through.

    Returns:
        dict with per_layer role classifications.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    results = []
    for l in range(n_layers):
        if l == 0:
            pre_key = 'blocks.0.hook_resid_pre'
        else:
            pre_key = f'blocks.{l-1}.hook_resid_post'
        post_key = f'blocks.{l}.hook_resid_post'

        if pre_key not in cache or post_key not in cache:
            continue

        pre = cache[pre_key][pos]
        post = cache[post_key][pos]
        delta = post - pre

        pre_norm = float(jnp.linalg.norm(pre))
        delta_norm = float(jnp.linalg.norm(delta))
        post_norm = float(jnp.linalg.norm(post))

        # Cosine between pre and delta
        cos_pre_delta = float(jnp.dot(pre, delta) / jnp.maximum(pre_norm * delta_norm, 1e-10))

        relative_change = delta_norm / max(pre_norm, 1e-10)

        # Classify
        if relative_change < 0.05:
            role = 'passthrough'
        elif abs(cos_pre_delta) > 0.5:
            role = 'refining'  # delta aligned with pre
        else:
            role = 'transforming'  # delta orthogonal to pre

        results.append({
            'layer': l,
            'role': role,
            'relative_change': relative_change,
            'pre_delta_alignment': cos_pre_delta,
        })

    return {'per_layer': results}
