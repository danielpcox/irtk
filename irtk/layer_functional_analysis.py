"""Layer functional analysis: characterize what each layer does.

Analyze each layer's functional role: information compression, prediction
refinement, representation transformation, and specialization.
"""

import jax
import jax.numpy as jnp


def layer_information_gain(model, tokens, position=-1):
    """How much does each layer reduce prediction uncertainty?

    Returns:
        dict with 'per_layer' entropy and KL divergence from previous layer.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    per_layer = []
    prev_probs = None
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        entropy = float(-jnp.sum(probs * log_probs))
        kl = 0.0
        if prev_probs is not None:
            kl = float(jnp.sum(probs * (jnp.log(probs + 1e-10) - jnp.log(prev_probs + 1e-10))))
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "kl_from_previous": kl,
        })
        prev_probs = probs
    return {"per_layer": per_layer}


def layer_transformation_magnitude(model, tokens):
    """How much does each layer transform the residual stream?

    Returns:
        dict with 'per_layer' delta norms and cosine with input.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        pre = cache[("resid_pre", layer)] if layer == 0 else cache[("resid_post", layer - 1)]
        post = cache[("resid_post", layer)]
        delta = post - pre
        delta_norm = float(jnp.mean(jnp.linalg.norm(delta, axis=-1)))
        pre_norm = float(jnp.mean(jnp.linalg.norm(pre, axis=-1)))
        # Mean cosine between pre and post
        cos_per_pos = jnp.sum(pre * post, axis=-1) / (
            jnp.linalg.norm(pre, axis=-1) * jnp.linalg.norm(post, axis=-1) + 1e-10
        )
        mean_cos = float(jnp.mean(cos_per_pos))
        per_layer.append({
            "layer": layer,
            "delta_norm": delta_norm,
            "relative_change": delta_norm / (pre_norm + 1e-10),
            "pre_post_cosine": mean_cos,
        })
    return {"per_layer": per_layer}


def layer_specialization_score(model, tokens):
    """How specialized is each layer (attn-dominant, mlp-dominant, balanced)?

    Returns:
        dict with 'per_layer' specialization metrics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)]  # [seq, d_model]
        mlp_out = cache[("mlp_out", layer)]  # [seq, d_model]
        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=-1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))
        total = attn_norm + mlp_norm + 1e-10
        attn_frac = attn_norm / total
        mlp_frac = mlp_norm / total
        # Balance: 0.5 = perfectly balanced, 0/1 = completely specialized
        balance = 1.0 - abs(attn_frac - mlp_frac)
        if attn_frac > 0.6:
            spec = "attn_dominant"
        elif mlp_frac > 0.6:
            spec = "mlp_dominant"
        else:
            spec = "balanced"
        per_layer.append({
            "layer": layer,
            "attn_fraction": attn_frac,
            "mlp_fraction": mlp_frac,
            "balance": balance,
            "specialization": spec,
        })
    return {"per_layer": per_layer}


def layer_prediction_contribution(model, tokens, position=-1):
    """How much does each layer change the top prediction?

    Returns:
        dict with 'per_layer' prediction changes and logit deltas.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        if layer == 0:
            pre_resid = cache[("resid_pre", 0)][position]
        else:
            pre_resid = cache[("resid_post", layer - 1)][position]
        post_resid = cache[("resid_post", layer)][position]
        pre_logits = pre_resid @ W_U + b_U
        post_logits = post_resid @ W_U + b_U
        pre_pred = int(jnp.argmax(pre_logits))
        post_pred = int(jnp.argmax(post_logits))
        logit_delta = float(jnp.linalg.norm(post_logits - pre_logits))
        per_layer.append({
            "layer": layer,
            "pre_prediction": pre_pred,
            "post_prediction": post_pred,
            "changed": pre_pred != post_pred,
            "logit_delta_norm": logit_delta,
        })
    return {"per_layer": per_layer}


def layer_functional_summary(model, tokens, position=-1):
    """Summary of layer functional analysis.

    Returns:
        dict with 'per_layer' combined metrics.
    """
    info = layer_information_gain(model, tokens, position=position)
    trans = layer_transformation_magnitude(model, tokens)
    spec = layer_specialization_score(model, tokens)
    return {
        "per_layer": [{
            "layer": i,
            "entropy": info["per_layer"][i]["entropy"],
            "delta_norm": trans["per_layer"][i]["delta_norm"],
            "specialization": spec["per_layer"][i]["specialization"],
            "balance": spec["per_layer"][i]["balance"],
        } for i in range(len(info["per_layer"]))],
    }
