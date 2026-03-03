"""Cross-component interaction: how attention and MLP interact within a layer.

Analyze whether attention and MLP reinforce or compete, their joint effect
on the residual, and information flow between them.
"""

import jax.numpy as jnp


def attn_mlp_cosine(model, tokens, layer=0, position=-1):
    """Cosine similarity between attention and MLP outputs.

    Positive = reinforcing, negative = competing.

    Returns:
        dict with 'cosine', 'attn_norm', 'mlp_norm', 'relationship'.
    """
    _, cache = model.run_with_cache(tokens)
    attn_out = cache[("attn_out", layer)][position]
    mlp_out = cache[("mlp_out", layer)][position]
    cos = float(jnp.dot(attn_out, mlp_out) / (
        jnp.linalg.norm(attn_out) * jnp.linalg.norm(mlp_out) + 1e-10))
    return {
        "cosine": cos,
        "attn_norm": float(jnp.linalg.norm(attn_out)),
        "mlp_norm": float(jnp.linalg.norm(mlp_out)),
        "relationship": "reinforcing" if cos > 0.1 else ("competing" if cos < -0.1 else "orthogonal"),
    }


def attn_mlp_logit_agreement(model, tokens, layer=0, position=-1, top_k=5):
    """Do attention and MLP promote the same tokens?

    Returns:
        dict with 'attn_promoted', 'mlp_promoted', 'overlap' count.
    """
    _, cache = model.run_with_cache(tokens)
    attn_out = cache[("attn_out", layer)][position]
    mlp_out = cache[("mlp_out", layer)][position]
    W_U = model.unembed.W_U
    attn_logits = attn_out @ W_U
    mlp_logits = mlp_out @ W_U
    attn_top = set(int(i) for i in jnp.argsort(-attn_logits)[:top_k])
    mlp_top = set(int(i) for i in jnp.argsort(-mlp_logits)[:top_k])
    overlap = attn_top & mlp_top
    return {
        "attn_promoted": sorted(attn_top),
        "mlp_promoted": sorted(mlp_top),
        "overlap": sorted(overlap),
        "n_overlap": len(overlap),
        "agreement_fraction": len(overlap) / top_k,
    }


def component_contribution_to_prediction(model, tokens, layer=0, position=-1):
    """How much does each component contribute to the final prediction logit?

    Returns:
        dict with attn and mlp contributions toward the predicted token.
    """
    _, cache = model.run_with_cache(tokens)
    attn_out = cache[("attn_out", layer)][position]
    mlp_out = cache[("mlp_out", layer)][position]
    resid_post = cache[("resid_post", len(model.blocks) - 1)][position]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    final_logits = resid_post @ W_U + b_U
    pred_token = int(jnp.argmax(final_logits))
    unembed_dir = W_U[:, pred_token]
    attn_contrib = float(jnp.dot(attn_out, unembed_dir))
    mlp_contrib = float(jnp.dot(mlp_out, unembed_dir))
    return {
        "predicted_token": pred_token,
        "attn_contribution": attn_contrib,
        "mlp_contribution": mlp_contrib,
        "total_contribution": attn_contrib + mlp_contrib,
    }


def residual_mid_analysis(model, tokens, layer=0, position=-1):
    """Analyze the residual stream between attention and MLP.

    How does the stream change after attention but before MLP?

    Returns:
        dict with norms and cosines at the mid-point.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)][position]
    resid_mid = cache[("resid_mid", layer)][position]
    resid_post = cache[("resid_post", layer)][position]
    pre_norm = float(jnp.linalg.norm(resid_pre))
    mid_norm = float(jnp.linalg.norm(resid_mid))
    post_norm = float(jnp.linalg.norm(resid_post))
    pre_mid_cos = float(jnp.dot(resid_pre, resid_mid) / (pre_norm * mid_norm + 1e-10))
    mid_post_cos = float(jnp.dot(resid_mid, resid_post) / (mid_norm * post_norm + 1e-10))
    return {
        "pre_norm": pre_norm,
        "mid_norm": mid_norm,
        "post_norm": post_norm,
        "pre_mid_cosine": pre_mid_cos,
        "mid_post_cosine": mid_post_cos,
    }


def cross_component_summary(model, tokens, position=-1):
    """Summary of cross-component interactions across all layers.

    Returns:
        dict with 'per_layer' list.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        cos = attn_mlp_cosine(model, tokens, layer=layer, position=position)
        agree = attn_mlp_logit_agreement(model, tokens, layer=layer, position=position, top_k=5)
        per_layer.append({
            "layer": layer,
            "cosine": cos["cosine"],
            "relationship": cos["relationship"],
            "agreement": agree["agreement_fraction"],
        })
    return {"per_layer": per_layer}
