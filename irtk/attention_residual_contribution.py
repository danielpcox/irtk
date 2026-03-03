"""Attention residual contribution: how attention outputs shape the residual stream."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def attention_residual_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                                   layer: int = 0) -> dict:
    """Cosine alignment between attention output and residual stream.

    High alignment means attention reinforces the existing representation;
    low alignment means attention adds orthogonal information.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]  # [seq, d_model]
    attn_out = cache[("attn_out", layer)]  # [seq, d_model]

    per_position = []
    for pos in range(resid_pre.shape[0]):
        r = resid_pre[pos]
        a = attn_out[pos]
        r_norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        a_norm = jnp.sqrt(jnp.sum(a ** 2)).clip(1e-8)
        cos = float(jnp.sum(r * a) / (r_norm * a_norm))
        per_position.append({
            "position": pos,
            "cosine": cos,
            "attn_norm": float(a_norm),
            "resid_norm": float(r_norm),
        })
    cosines = [p["cosine"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_alignment": sum(cosines) / len(cosines),
        "is_reinforcing": sum(cosines) / len(cosines) > 0.3,
    }


def per_head_residual_contribution(model: HookedTransformer, tokens: jnp.ndarray,
                                     layer: int = 0) -> dict:
    """Each head's contribution magnitude and direction relative to residual.

    Shows which heads are dominant writers to the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]  # [seq, d_model]
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for head in range(model.cfg.n_heads):
        # Head output: z[:, head, :] @ W_O[head]
        head_out = z[:, head, :] @ W_O[head]  # [seq, d_model]
        head_norm = float(jnp.mean(jnp.sqrt(jnp.sum(head_out ** 2, axis=-1))))

        # Mean alignment with residual
        cosines = []
        for pos in range(resid_pre.shape[0]):
            r_norm = jnp.sqrt(jnp.sum(resid_pre[pos] ** 2)).clip(1e-8)
            h_norm = jnp.sqrt(jnp.sum(head_out[pos] ** 2)).clip(1e-8)
            cos = float(jnp.sum(resid_pre[pos] * head_out[pos]) / (r_norm * h_norm))
            cosines.append(cos)

        per_head.append({
            "head": int(head),
            "mean_norm": head_norm,
            "mean_alignment": sum(cosines) / len(cosines),
        })

    norms = [h["mean_norm"] for h in per_head]
    total = sum(norms)
    for h, n in zip(per_head, norms):
        h["fraction"] = n / max(total, 1e-8)

    return {
        "layer": layer,
        "per_head": per_head,
        "dominant_head": max(range(len(per_head)), key=lambda i: per_head[i]["mean_norm"]),
    }


def attention_update_magnitude(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0) -> dict:
    """Relative magnitude of attention update vs residual stream.

    Shows how much the attention layer modifies the representation.
    """
    _, cache = model.run_with_cache(tokens)
    resid_pre = cache[("resid_pre", layer)]
    attn_out = cache[("attn_out", layer)]

    per_position = []
    for pos in range(resid_pre.shape[0]):
        r_norm = float(jnp.sqrt(jnp.sum(resid_pre[pos] ** 2)))
        a_norm = float(jnp.sqrt(jnp.sum(attn_out[pos] ** 2)))
        ratio = a_norm / max(r_norm, 1e-8)
        per_position.append({
            "position": pos,
            "update_ratio": ratio,
            "resid_norm": r_norm,
            "attn_norm": a_norm,
        })
    ratios = [p["update_ratio"] for p in per_position]
    return {
        "layer": layer,
        "per_position": per_position,
        "mean_update_ratio": sum(ratios) / len(ratios),
        "is_large_update": sum(ratios) / len(ratios) > 0.5,
    }


def attention_direction_consistency(model: HookedTransformer, tokens: jnp.ndarray,
                                      layer: int = 0) -> dict:
    """Consistency of attention output direction across positions.

    High consistency means attention writes the same direction everywhere;
    low consistency means position-specific outputs.
    """
    _, cache = model.run_with_cache(tokens)
    attn_out = cache[("attn_out", layer)]  # [seq, d_model]
    seq_len = attn_out.shape[0]

    norms = jnp.sqrt(jnp.sum(attn_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = attn_out / norms
    sim = normed @ normed.T
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    return {
        "layer": layer,
        "mean_direction_similarity": mean_sim,
        "is_consistent": mean_sim > 0.5,
        "seq_len": int(seq_len),
    }


def attention_residual_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer attention-residual contribution summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        align = attention_residual_alignment(model, tokens, layer)
        mag = attention_update_magnitude(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_alignment": align["mean_alignment"],
            "is_reinforcing": align["is_reinforcing"],
            "mean_update_ratio": mag["mean_update_ratio"],
        })
    return {"per_layer": per_layer}
