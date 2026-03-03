"""Logit formation analysis: how final logits are assembled."""

import jax
import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def logit_buildup_trajectory(model: HookedTransformer, tokens: jnp.ndarray,
                                position: int = -1, token_id: int = 0) -> dict:
    """Track how a specific token's logit builds up through layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U
    target_dir = W_U[:, token_id]

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logit = float(jnp.sum(resid * target_dir) + b_U[token_id])

        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        attn_contrib = float(jnp.sum(attn_out * target_dir))
        mlp_contrib = float(jnp.sum(mlp_out * target_dir))

        per_layer.append({
            "layer": layer,
            "logit": logit,
            "attn_contribution": attn_contrib,
            "mlp_contribution": mlp_contrib,
        })

    return {
        "position": position,
        "token_id": token_id,
        "per_layer": per_layer,
        "final_logit": per_layer[-1]["logit"] if per_layer else 0.0,
    }


def top_logit_contributors(model: HookedTransformer, tokens: jnp.ndarray,
                              position: int = -1, top_k: int = 5) -> dict:
    """Find which layers contribute most to the top predicted token's logit."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)][position]
    final_logits = final_resid @ W_U + b_U
    top_token = int(jnp.argmax(final_logits))
    target_dir = W_U[:, top_token]

    contributions = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        contributions.append({
            "layer": layer,
            "component": "attn",
            "contribution": float(jnp.sum(attn_out * target_dir)),
        })
        contributions.append({
            "layer": layer,
            "component": "mlp",
            "contribution": float(jnp.sum(mlp_out * target_dir)),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return {
        "position": position,
        "top_token": top_token,
        "top_contributors": contributions[:top_k],
        "total_contributions": len(contributions),
    }


def logit_competition_analysis(model: HookedTransformer, tokens: jnp.ndarray,
                                  position: int = -1, top_k: int = 3) -> dict:
    """How the top-k tokens compete for highest logit through layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)][position]
    final_logits = final_resid @ W_U + b_U
    top_tokens = [int(t) for t in jnp.argsort(-final_logits)[:top_k]]

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        token_logits = {t: float(logits[t]) for t in top_tokens}
        leader = max(top_tokens, key=lambda t: token_logits[t])
        per_layer.append({
            "layer": layer,
            "token_logits": token_logits,
            "leader": leader,
        })

    leader_changes = sum(1 for i in range(len(per_layer) - 1)
                        if per_layer[i]["leader"] != per_layer[i + 1]["leader"])
    return {
        "position": position,
        "tracked_tokens": top_tokens,
        "per_layer": per_layer,
        "leader_changes": leader_changes,
    }


def embedding_logit_bias(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1, top_k: int = 5) -> dict:
    """What logit bias does the embedding layer provide?"""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    embed = cache["hook_embed"][position]
    pos_embed = cache["hook_pos_embed"][position]
    combined = embed + pos_embed

    logits = combined @ W_U + b_U
    top_indices = jnp.argsort(-logits)[:top_k]
    top_predictions = [(int(idx), float(logits[idx])) for idx in top_indices]

    return {
        "position": position,
        "top_predictions": top_predictions,
        "embed_norm": float(jnp.sqrt(jnp.sum(embed ** 2))),
        "pos_embed_norm": float(jnp.sqrt(jnp.sum(pos_embed ** 2))),
    }


def logit_formation_summary(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """Combined logit formation summary."""
    if position < 0:
        position = len(tokens) + position
    contrib = top_logit_contributors(model, tokens, position)
    comp = logit_competition_analysis(model, tokens, position)
    return {
        "position": position,
        "top_token": contrib["top_token"],
        "top_contributor": contrib["top_contributors"][0] if contrib["top_contributors"] else None,
        "leader_changes": comp["leader_changes"],
    }
