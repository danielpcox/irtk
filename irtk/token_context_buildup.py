"""Token context buildup: how context accumulates in token representations."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def context_accumulation_rate(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """Measure how much a token's representation changes at each layer.

    Large changes indicate the token is accumulating new context;
    small changes mean the representation has stabilized.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid_pre = cache[("resid_pre", layer)]
        resid_post = cache[("resid_post", layer)]
        pre = resid_pre[position]
        post = resid_post[position]
        delta = post - pre
        delta_norm = float(jnp.sqrt(jnp.sum(delta ** 2)))
        pre_norm = float(jnp.sqrt(jnp.sum(pre ** 2)).clip(1e-8))
        # Cosine between pre and post
        post_norm = float(jnp.sqrt(jnp.sum(post ** 2)).clip(1e-8))
        cos = float(jnp.sum(pre * post) / (pre_norm * post_norm))
        per_layer.append({
            "layer": layer,
            "update_norm": delta_norm,
            "relative_update": delta_norm / pre_norm,
            "pre_post_cosine": cos,
        })
    updates = [p["relative_update"] for p in per_layer]
    return {
        "per_layer": per_layer,
        "total_relative_update": sum(updates),
        "update_trend": "decreasing" if updates[-1] < updates[0] * 0.5 else
                       "increasing" if updates[-1] > updates[0] * 1.5 else "stable",
    }


def context_source_attribution(model: HookedTransformer, tokens: jnp.ndarray,
                                position: int = -1) -> dict:
    """Attribute context accumulation to attention vs MLP at each layer."""
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]
        attn_norm = float(jnp.sqrt(jnp.sum(attn_out[position] ** 2)))
        mlp_norm = float(jnp.sqrt(jnp.sum(mlp_out[position] ** 2)))
        total = attn_norm + mlp_norm
        per_layer.append({
            "layer": layer,
            "attn_contribution": attn_norm,
            "mlp_contribution": mlp_norm,
            "attn_fraction": attn_norm / max(total, 1e-8),
            "mlp_fraction": mlp_norm / max(total, 1e-8),
        })
    return {
        "per_layer": per_layer,
        "mean_attn_fraction": sum(p["attn_fraction"] for p in per_layer) / len(per_layer),
    }


def position_context_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = -1) -> dict:
    """Measure how different positions' representations diverge from each other.

    Early layers may have similar representations; later layers should
    be more position-specific as context accumulates.
    """
    if layer < 0:
        layer = model.cfg.n_layers + layer
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    seq_len = resid.shape[0]

    norms = jnp.sqrt(jnp.sum(resid ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = resid / norms
    sim = normed @ normed.T  # [seq, seq]
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    return {
        "layer": layer,
        "mean_pairwise_similarity": mean_sim,
        "is_diverse": mean_sim < 0.5,
        "n_positions": int(seq_len),
    }


def embedding_distance_tracking(model: HookedTransformer, tokens: jnp.ndarray,
                                 position: int = -1) -> dict:
    """Track how far the representation moves from the initial embedding.

    Shows the journey from static token embedding to contextualized
    representation.
    """
    _, cache = model.run_with_cache(tokens)
    embed = cache["hook_embed"]  # [seq, d_model]
    initial = embed[position]
    initial_norm = jnp.sqrt(jnp.sum(initial ** 2)).clip(1e-8)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        current = resid[position]
        current_norm = jnp.sqrt(jnp.sum(current ** 2)).clip(1e-8)
        distance = float(jnp.sqrt(jnp.sum((current - initial) ** 2)))
        cosine = float(jnp.sum(initial * current) / (initial_norm * current_norm))
        per_layer.append({
            "layer": layer,
            "distance_from_embed": distance,
            "cosine_to_embed": cosine,
            "norm": float(current_norm),
        })
    return {
        "per_layer": per_layer,
        "final_distance": per_layer[-1]["distance_from_embed"],
        "final_cosine": per_layer[-1]["cosine_to_embed"],
    }


def context_buildup_summary(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Combined context buildup analysis."""
    rate = context_accumulation_rate(model, tokens, position)
    source = context_source_attribution(model, tokens, position)
    distance = embedding_distance_tracking(model, tokens, position)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        per_layer.append({
            "layer": layer,
            "relative_update": rate["per_layer"][layer]["relative_update"],
            "attn_fraction": source["per_layer"][layer]["attn_fraction"],
            "distance_from_embed": distance["per_layer"][layer]["distance_from_embed"],
            "cosine_to_embed": distance["per_layer"][layer]["cosine_to_embed"],
        })
    return {
        "per_layer": per_layer,
        "update_trend": rate["update_trend"],
        "final_distance": distance["final_distance"],
        "final_cosine": distance["final_cosine"],
        "mean_attn_fraction": source["mean_attn_fraction"],
    }
