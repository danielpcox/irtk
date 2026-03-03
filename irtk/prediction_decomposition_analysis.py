"""Prediction decomposition analysis: decompose final predictions by component.

Break down the model's final logit vector into contributions from
each layer's attention and MLP, showing which components drive each prediction.
"""

import jax
import jax.numpy as jnp


def logit_contribution_by_layer(model, tokens, position=-1, top_k=5):
    """Decompose the final logit into per-layer contributions.

    For each layer, computes how much the attention and MLP outputs
    contribute to the final logit via the unembedding.

    Returns:
        dict with 'per_layer' list of contribution dicts,
        'top_tokens' top predicted tokens.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]
    n_layers = len(model.blocks)

    # Final prediction
    final_resid = cache[("resid_post", n_layers - 1)]
    final_logits = final_resid[position] @ W_U + b_U
    top_tokens = jnp.argsort(-final_logits)[:top_k]

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]  # [d_model]
        mlp_out = cache[("mlp_out", layer)][position]  # [d_model]
        attn_logits = attn_out @ W_U  # [d_vocab]
        mlp_logits = mlp_out @ W_U  # [d_vocab]
        per_layer.append({
            "layer": layer,
            "attn_contribution": float(jnp.sum(jnp.abs(attn_logits))),
            "mlp_contribution": float(jnp.sum(jnp.abs(mlp_logits))),
            "attn_top_token_logits": {int(t): float(attn_logits[t]) for t in top_tokens},
            "mlp_top_token_logits": {int(t): float(mlp_logits[t]) for t in top_tokens},
        })
    return {
        "per_layer": per_layer,
        "top_tokens": [int(t) for t in top_tokens],
        "final_logits_top": {int(t): float(final_logits[t]) for t in top_tokens},
    }


def direct_logit_attribution(model, tokens, position=-1, target_token=None):
    """Which components contribute most to a specific token's logit?

    If target_token is None, uses the argmax prediction.

    Returns:
        dict with 'target_token', 'per_component' list ranked by contribution.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)

    if target_token is None:
        final_resid = cache[("resid_post", n_layers - 1)]
        final_logits = final_resid[position] @ W_U + b_U
        target_token = int(jnp.argmax(final_logits))

    unembed_dir = W_U[:, target_token]  # [d_model]
    components = []

    # Embedding contribution
    embed_resid = cache[("resid_pre", 0)][position]
    components.append({
        "component": "embed",
        "layer": -1,
        "contribution": float(jnp.dot(embed_resid, unembed_dir)),
    })

    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        components.append({
            "component": f"attn_{layer}",
            "layer": layer,
            "contribution": float(jnp.dot(attn_out, unembed_dir)),
        })
        components.append({
            "component": f"mlp_{layer}",
            "layer": layer,
            "contribution": float(jnp.dot(mlp_out, unembed_dir)),
        })

    # Add bias
    components.append({
        "component": "unembed_bias",
        "layer": -1,
        "contribution": float(b_U[target_token]),
    })

    components.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {
        "target_token": target_token,
        "per_component": components,
        "total_logit": sum(c["contribution"] for c in components),
    }


def prediction_entropy_decomposition(model, tokens, position=-1):
    """How does each layer affect the prediction entropy?

    Returns:
        dict with 'per_layer' entropy changes, 'total_entropy_reduction'.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    per_layer = []
    prev_entropy = None
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        entropy = float(-jnp.sum(probs * log_probs))
        change = entropy - prev_entropy if prev_entropy is not None else 0.0
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "entropy_change": change,
        })
        prev_entropy = entropy
    total_reduction = per_layer[0]["entropy"] - per_layer[-1]["entropy"] if len(per_layer) > 1 else 0.0
    return {
        "per_layer": per_layer,
        "total_entropy_reduction": total_reduction,
    }


def component_prediction_agreement(model, tokens, position=-1, top_k=5):
    """Do attention and MLP within a layer agree on what to predict?

    Returns:
        dict with 'per_layer' agreement scores.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        attn_logits = attn_out @ W_U
        mlp_logits = mlp_out @ W_U
        attn_top = set(int(t) for t in jnp.argsort(-attn_logits)[:top_k])
        mlp_top = set(int(t) for t in jnp.argsort(-mlp_logits)[:top_k])
        overlap = len(attn_top & mlp_top) / top_k
        # Cosine of logit vectors
        cos = float(jnp.dot(attn_logits, mlp_logits) / (
            jnp.linalg.norm(attn_logits) * jnp.linalg.norm(mlp_logits) + 1e-10
        ))
        per_layer.append({
            "layer": layer,
            "top_k_overlap": overlap,
            "logit_cosine": cos,
        })
    return {"per_layer": per_layer, "top_k": top_k}


def prediction_decomposition_summary(model, tokens, position=-1):
    """Summary of prediction decomposition.

    Returns:
        dict with key metrics from all decomposition analyses.
    """
    contrib = logit_contribution_by_layer(model, tokens, position=position)
    entropy = prediction_entropy_decomposition(model, tokens, position=position)
    agree = component_prediction_agreement(model, tokens, position=position)
    return {
        "top_tokens": contrib["top_tokens"],
        "total_entropy_reduction": entropy["total_entropy_reduction"],
        "per_layer": [{
            "layer": i,
            "attn_contribution": contrib["per_layer"][i]["attn_contribution"],
            "mlp_contribution": contrib["per_layer"][i]["mlp_contribution"],
            "entropy": entropy["per_layer"][i]["entropy"],
            "agreement": agree["per_layer"][i]["top_k_overlap"],
        } for i in range(len(contrib["per_layer"]))],
    }
