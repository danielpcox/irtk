"""Attention pattern evolution: track how attention patterns change across layers.

Analyze attention stability, pattern shifts, head agreement evolution,
and entropy trends across the network depth.
"""

import jax
import jax.numpy as jnp


def attention_stability_across_layers(model, tokens, head=0):
    """Track how stable attention patterns are for a specific head index across layers.

    Measures cosine similarity between the same head's pattern at adjacent layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    patterns = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        patterns.append(cache[key][head])  # [seq, seq]

    per_pair = []
    for i in range(len(patterns) - 1):
        a = patterns[i].reshape(-1)
        b = patterns[i + 1].reshape(-1)
        cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10))
        per_pair.append({
            "layer_from": i,
            "layer_to": i + 1,
            "cosine_similarity": cos,
        })

    mean_stability = sum(p["cosine_similarity"] for p in per_pair) / max(len(per_pair), 1)

    return {
        "per_pair": per_pair,
        "head": head,
        "mean_stability": mean_stability,
        "is_stable": mean_stability > 0.7,
    }


def attention_focus_evolution(model, tokens, position=-1):
    """Track where a specific position focuses attention across layers.

    Shows the top attended positions at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        avg_attn = jnp.mean(pattern[:, position, :], axis=0)  # [seq]

        top_k = min(3, len(tokens))
        top_indices = jnp.argsort(avg_attn)[-top_k:][::-1]
        focus = []
        for idx in top_indices:
            focus.append({
                "position": int(idx),
                "token_id": int(tokens[int(idx)]),
                "attention": float(avg_attn[int(idx)]),
            })

        entropy = -float(jnp.sum(avg_attn * jnp.log(avg_attn + 1e-10)))

        per_layer.append({
            "layer": layer,
            "top_focus": focus,
            "entropy": entropy,
            "top_position": int(top_indices[0]),
        })

    return {
        "per_layer": per_layer,
        "query_position": position,
        "focus_shifts": sum(1 for i in range(len(per_layer) - 1)
                          if per_layer[i]["top_position"] != per_layer[i + 1]["top_position"]),
    }


def head_agreement_evolution(model, tokens, position=-1):
    """Track how much heads agree on where to attend across layers.

    High agreement = heads attend to the same positions; low = diverse focus.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]

        # Get each head's attention for this query position
        head_attns = pattern[:, position, :]  # [n_heads, seq]

        # Pairwise cosine between heads
        norms = jnp.linalg.norm(head_attns, axis=-1, keepdims=True) + 1e-10
        normed = head_attns / norms
        sim = normed @ normed.T  # [n_heads, n_heads]

        n = n_heads
        off_diag_mask = 1.0 - jnp.eye(n)
        mean_agreement = float(jnp.sum(sim * off_diag_mask) / (jnp.sum(off_diag_mask) + 1e-10))

        per_layer.append({
            "layer": layer,
            "mean_head_agreement": mean_agreement,
            "heads_agree": mean_agreement > 0.7,
        })

    return {
        "per_layer": per_layer,
        "query_position": position,
        "agreement_trend": "increasing" if per_layer and per_layer[-1]["mean_head_agreement"] > per_layer[0]["mean_head_agreement"] else "decreasing",
    }


def attention_entropy_evolution(model, tokens):
    """Track attention entropy across all layers and heads.

    Shows whether attention sharpens or broadens through the network.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        entropy = -jnp.sum(pattern * jnp.log(pattern + 1e-10), axis=-1)  # [n_heads, seq]
        mean_entropy = float(jnp.mean(entropy))
        per_head = [float(jnp.mean(entropy[h])) for h in range(n_heads)]

        per_layer.append({
            "layer": layer,
            "mean_entropy": mean_entropy,
            "per_head_entropy": per_head,
            "min_entropy_head": int(jnp.argmin(jnp.array(per_head))),
            "max_entropy_head": int(jnp.argmax(jnp.array(per_head))),
        })

    sharpening = per_layer[-1]["mean_entropy"] < per_layer[0]["mean_entropy"] if per_layer else False

    return {
        "per_layer": per_layer,
        "sharpens": sharpening,
        "entropy_range": per_layer[0]["mean_entropy"] - per_layer[-1]["mean_entropy"] if per_layer else 0.0,
    }


def attention_evolution_summary(model, tokens):
    """Cross-layer summary of attention pattern evolution.

    Combines stability, focus, agreement, and entropy metrics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    per_layer = []
    prev_avg = None
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]
        avg_attn = jnp.mean(pattern, axis=0)  # [seq, seq]

        # Entropy
        entropy = -jnp.sum(avg_attn * jnp.log(avg_attn + 1e-10), axis=-1)
        mean_entropy = float(jnp.mean(entropy))

        # Stability from previous
        stability = 0.0
        if prev_avg is not None:
            a = prev_avg.reshape(-1)
            b = avg_attn.reshape(-1)
            stability = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10))

        # Self-attention
        self_attn = float(jnp.mean(jnp.diag(avg_attn)))

        per_layer.append({
            "layer": layer,
            "mean_entropy": mean_entropy,
            "stability_from_prev": stability,
            "self_attention": self_attn,
        })
        prev_avg = avg_attn

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
    }
