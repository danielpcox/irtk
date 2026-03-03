"""Attention value routing analysis: where values come from and go."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def value_source_decomposition(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0, head: int = 0,
                                  position: int = -1) -> dict:
    """Decompose the output of a head by source position.

    For a given destination position, shows how much each source contributes.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    pattern = cache[("pattern", layer)][head]  # [seq_q, seq_k]
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]

    resid = cache[("resid_pre", layer)]  # [seq, d_model]

    per_source = []
    for src in range(len(tokens)):
        attn_weight = float(pattern[position, src])
        value = resid[src] @ W_V  # [d_head]
        output = value @ W_O  # [d_model]
        contribution_norm = float(jnp.sqrt(jnp.sum(output ** 2))) * attn_weight
        per_source.append({
            "source": src,
            "attention_weight": attn_weight,
            "contribution_norm": contribution_norm,
        })

    return {
        "layer": layer,
        "head": head,
        "position": position,
        "per_source": per_source,
        "dominant_source": max(range(len(per_source)),
                              key=lambda i: per_source[i]["contribution_norm"]),
    }


def value_diversity_per_head(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = 0) -> dict:
    """How diverse are the value vectors across positions for each head?"""
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_pre", layer)]  # [seq, d_model]
    W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]

    per_head = []
    for head in range(model.cfg.n_heads):
        values = resid @ W_V[head]  # [seq, d_head]
        norms = jnp.sqrt(jnp.sum(values ** 2, axis=-1, keepdims=True)).clip(1e-8)
        normed = values / norms
        sim = normed @ normed.T
        mask = 1.0 - jnp.eye(len(tokens))
        mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))
        per_head.append({
            "head": int(head),
            "mean_similarity": mean_sim,
            "diversity": 1.0 - mean_sim,
        })

    return {
        "layer": layer,
        "per_head": per_head,
        "mean_diversity": sum(h["diversity"] for h in per_head) / len(per_head),
    }


def attention_routing_entropy(model: HookedTransformer, tokens: jnp.ndarray,
                                 layer: int = 0) -> dict:
    """Entropy of attention routing: how spread out is the value routing?"""
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq_q, seq_k]

    per_head = []
    for head in range(model.cfg.n_heads):
        entropies = []
        for pos in range(len(tokens)):
            p = pattern[head, pos, :pos + 1]
            p = p.clip(1e-10)
            h = float(-jnp.sum(p * jnp.log(p)))
            entropies.append(h)
        mean_entropy = sum(entropies) / len(entropies)
        max_entropy = float(jnp.log(jnp.array(len(tokens), dtype=jnp.float32)))
        per_head.append({
            "head": int(head),
            "mean_entropy": mean_entropy,
            "normalized_entropy": mean_entropy / max(max_entropy, 1e-8),
            "is_focused": mean_entropy < max_entropy * 0.3,
        })

    return {
        "layer": layer,
        "per_head": per_head,
        "n_focused": sum(1 for h in per_head if h["is_focused"]),
    }


def value_output_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0, position: int = -1) -> dict:
    """Alignment between value vectors and actual head output."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
    resid = cache[("resid_pre", layer)]  # [seq, d_model]

    per_head = []
    for head in range(model.cfg.n_heads):
        actual_out = z[position, head, :] @ W_O[head]  # [d_model]
        mean_value = jnp.mean(resid @ W_V[head], axis=0) @ W_O[head]  # [d_model]

        a_norm = jnp.sqrt(jnp.sum(actual_out ** 2)).clip(1e-8)
        m_norm = jnp.sqrt(jnp.sum(mean_value ** 2)).clip(1e-8)
        cos = float(jnp.sum(actual_out * mean_value) / (a_norm * m_norm))

        per_head.append({
            "head": int(head),
            "cosine": cos,
            "actual_norm": float(a_norm),
            "mean_value_norm": float(m_norm),
        })

    return {
        "layer": layer,
        "position": position,
        "per_head": per_head,
    }


def value_routing_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer value routing summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        div = value_diversity_per_head(model, tokens, layer)
        ent = attention_routing_entropy(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_diversity": div["mean_diversity"],
            "n_focused": ent["n_focused"],
        })
    return {"per_layer": per_layer}
