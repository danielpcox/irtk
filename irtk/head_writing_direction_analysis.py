"""Head writing direction analysis: what directions each head writes."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def head_writing_directions(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Principal writing directions for each head via SVD.

    Shows the dominant directions each head uses to write to the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[:, head, :] @ W_O[head]  # [seq, d_model]
        svs = jnp.linalg.svd(head_out, compute_uv=False)
        svs_norm = svs / jnp.sum(svs).clip(1e-8)
        eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

        per_head.append({
            "head": int(head),
            "effective_rank": eff_rank,
            "top_sv": float(svs[0]),
            "sv_concentration": float(svs[0] / jnp.sum(svs).clip(1e-8)),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_rank": sum(h["effective_rank"] for h in per_head) / len(per_head),
    }


def head_unembed_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0, top_k: int = 5) -> dict:
    """Alignment of head writing directions with unembedding directions.

    Shows which tokens each head's output promotes.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]
    W_O = model.blocks[layer].attn.W_O
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[:, head, :] @ W_O[head]  # [seq, d_model]
        mean_out = jnp.mean(head_out, axis=0)  # [d_model]
        logits = mean_out @ W_U  # [d_vocab]

        top_indices = jnp.argsort(-logits)[:top_k]
        top_tokens = [int(idx) for idx in top_indices]
        top_logits = [float(logits[idx]) for idx in top_indices]

        per_head.append({
            "head": int(head),
            "top_tokens": top_tokens,
            "top_logits": top_logits,
        })
    return {
        "layer": layer,
        "per_head": per_head,
    }


def head_direction_consistency(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0) -> dict:
    """How consistent are head writing directions across positions?

    High consistency = head writes same direction everywhere.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]
    W_O = model.blocks[layer].attn.W_O

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[:, head, :] @ W_O[head]  # [seq, d_model]
        seq_len = head_out.shape[0]
        norms = jnp.sqrt(jnp.sum(head_out ** 2, axis=-1, keepdims=True)).clip(1e-8)
        normed = head_out / norms
        sim = normed @ normed.T
        mask = 1.0 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

        per_head.append({
            "head": int(head),
            "direction_consistency": mean_sim,
            "is_consistent": mean_sim > 0.5,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "n_consistent": sum(1 for h in per_head if h["is_consistent"]),
    }


def head_writing_magnitude(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0) -> dict:
    """Writing magnitude per head: how much does each head contribute?"""
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]
    W_O = model.blocks[layer].attn.W_O

    per_head = []
    total_norm = 0.0
    for head in range(model.cfg.n_heads):
        head_out = z[:, head, :] @ W_O[head]
        mean_norm = float(jnp.mean(jnp.sqrt(jnp.sum(head_out ** 2, axis=-1))))
        per_head.append({
            "head": int(head),
            "mean_norm": mean_norm,
        })
        total_norm += mean_norm

    for h in per_head:
        h["fraction"] = h["mean_norm"] / max(total_norm, 1e-8)

    return {
        "layer": layer,
        "per_head": per_head,
        "dominant_head": max(range(len(per_head)), key=lambda i: per_head[i]["mean_norm"]),
    }


def head_writing_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer head writing direction summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        dirs = head_writing_directions(model, tokens, layer)
        cons = head_direction_consistency(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_rank": dirs["mean_rank"],
            "n_consistent": cons["n_consistent"],
        })
    return {"per_layer": per_layer}
