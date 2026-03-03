"""Unembed projection analysis: how components project through W_U."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def residual_unembed_trajectory(model: HookedTransformer, tokens: jnp.ndarray,
                                 position: int = -1, top_k: int = 5) -> dict:
    """Track top-predicted tokens at each layer through unembedding.

    Shows how the prediction evolves as the residual stream grows.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        logits = resid[position] @ W_U + b_U
        top_ids = jnp.argsort(logits)[::-1][:top_k]
        top_tokens = []
        for idx in top_ids:
            top_tokens.append({
                "token_id": int(idx),
                "logit": float(logits[idx]),
            })
        per_layer.append({
            "layer": layer,
            "top_prediction": int(top_ids[0]),
            "top_logit": float(logits[top_ids[0]]),
            "top_tokens": top_tokens,
        })
    return {
        "per_layer": per_layer,
        "final_prediction": per_layer[-1]["top_prediction"],
    }


def component_unembed_contribution(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0, position: int = -1,
                                    target_token: int = None) -> dict:
    """How much each component (attn, MLP) contributes to a target token's logit.

    Decomposes the logit into attention and MLP contributions.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    if target_token is None:
        final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
        final_logits = final_resid[position] @ W_U + b_U
        target_token = int(jnp.argmax(final_logits))

    attn_out = cache[("attn_out", layer)]
    mlp_out = cache[("mlp_out", layer)]

    attn_logit = float((attn_out[position] @ W_U)[target_token])
    mlp_logit = float((mlp_out[position] @ W_U)[target_token])

    return {
        "layer": layer,
        "target_token": target_token,
        "attn_contribution": attn_logit,
        "mlp_contribution": mlp_logit,
        "total_contribution": attn_logit + mlp_logit,
        "dominant": "attention" if abs(attn_logit) > abs(mlp_logit) else "mlp",
    }


def unembed_alignment_per_head(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0, position: int = -1,
                                target_token: int = None) -> dict:
    """Per-head contribution to a target token's logit via unembedding."""
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    if target_token is None:
        final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
        final_logits = final_resid[position] @ W_U + b_U
        target_token = int(jnp.argmax(final_logits))

    z = cache[("z", layer)]
    W_O = model.blocks[layer].attn.W_O
    unembed_dir = W_U[:, target_token]  # [d_model]

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[position, head, :] @ W_O[head]  # [d_model]
        logit_contribution = float(jnp.sum(head_out * unembed_dir))
        per_head.append({
            "head": int(head),
            "logit_contribution": logit_contribution,
            "output_norm": float(jnp.sqrt(jnp.sum(head_out ** 2))),
        })
    per_head.sort(key=lambda h: abs(h["logit_contribution"]), reverse=True)
    return {
        "layer": layer,
        "target_token": target_token,
        "per_head": per_head,
        "top_head": per_head[0]["head"],
    }


def unembed_direction_stability(model: HookedTransformer, tokens: jnp.ndarray,
                                 position: int = -1) -> dict:
    """How stable is the residual's alignment with the top-predicted unembed direction?

    Measures cosine between residual and the unembed direction of the
    final predicted token at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    final_logits = final_resid[position] @ W_U + b_U
    target = int(jnp.argmax(final_logits))
    unembed_dir = W_U[:, target]
    unembed_norm = jnp.sqrt(jnp.sum(unembed_dir ** 2)).clip(1e-8)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        r = resid[position]
        r_norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        cos = float(jnp.sum(r * unembed_dir) / (r_norm * unembed_norm))
        per_layer.append({
            "layer": layer,
            "cosine_to_unembed": cos,
        })
    return {
        "target_token": target,
        "per_layer": per_layer,
        "final_alignment": per_layer[-1]["cosine_to_unembed"],
    }


def unembed_projection_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                position: int = -1) -> dict:
    """Combined unembedding projection analysis."""
    traj = residual_unembed_trajectory(model, tokens, position, top_k=1)
    stab = unembed_direction_stability(model, tokens, position)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        per_layer.append({
            "layer": layer,
            "top_prediction": traj["per_layer"][layer]["top_prediction"],
            "top_logit": traj["per_layer"][layer]["top_logit"],
            "cosine_to_final_unembed": stab["per_layer"][layer]["cosine_to_unembed"],
        })
    return {
        "final_prediction": traj["final_prediction"],
        "per_layer": per_layer,
    }
