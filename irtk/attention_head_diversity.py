"""Attention head diversity: how different heads within a layer behave."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def pattern_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                       layer: int = 0) -> dict:
    """Measure diversity of attention patterns across heads in a layer.

    High diversity means heads attend to different positions;
    low diversity suggests redundancy.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]

    # Flatten patterns and compute pairwise cosine
    flat = patterns.reshape(n_heads, -1)  # [n_heads, seq*seq]
    norms = jnp.sqrt(jnp.sum(flat ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = flat / norms
    sim = normed @ normed.T  # [n_heads, n_heads]

    mask = 1.0 - jnp.eye(n_heads)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                "head_a": int(i),
                "head_b": int(j),
                "similarity": float(sim[i, j]),
            })
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    return {
        "layer": layer,
        "mean_pattern_similarity": mean_sim,
        "is_diverse": mean_sim < 0.5,
        "head_pairs": pairs,
    }


def output_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                      layer: int = 0, position: int = -1) -> dict:
    """Measure diversity of head outputs at a specific position.

    Diverse outputs mean heads contribute different information;
    similar outputs suggest redundancy.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    outputs = []
    for head in range(model.cfg.n_heads):
        out = z[position, head, :] @ W_O[head]  # [d_model]
        outputs.append(out)
    outputs = jnp.stack(outputs)  # [n_heads, d_model]

    norms = jnp.sqrt(jnp.sum(outputs ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = outputs / norms
    sim = normed @ normed.T
    mask = 1.0 - jnp.eye(model.cfg.n_heads)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    return {
        "layer": layer,
        "position": int(position % z.shape[0]),
        "mean_output_similarity": mean_sim,
        "is_diverse": mean_sim < 0.3,
    }


def entropy_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                       layer: int = 0) -> dict:
    """Compare attention entropy across heads.

    Some heads may be sharp (low entropy) while others are diffuse.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = patterns[head]  # [seq, seq]
        # Mean entropy across query positions
        entropy = -jnp.sum(p * jnp.log(p.clip(1e-10)), axis=-1)
        mean_entropy = float(jnp.mean(entropy))
        per_head.append({
            "head": int(head),
            "mean_entropy": mean_entropy,
            "max_entropy": float(jnp.max(entropy)),
            "min_entropy": float(jnp.min(entropy)),
        })
    entropies = [h["mean_entropy"] for h in per_head]
    entropy_range = max(entropies) - min(entropies)
    return {
        "layer": layer,
        "per_head": per_head,
        "entropy_range": entropy_range,
        "is_entropy_diverse": entropy_range > 0.5,
    }


def attention_focus_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Compare where each head focuses (top-attended position per query).

    Diverse focus means heads attend to different positions.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = patterns.shape[1]

    # For each head, get the argmax position for each query
    focus = jnp.argmax(patterns, axis=-1)  # [n_heads, seq]

    # Count head agreement: for each query, how many heads agree on focus
    agreements = []
    for pos in range(seq_len):
        head_focuses = focus[:, pos]  # [n_heads]
        unique_focuses = len(set(int(f) for f in head_focuses))
        agreements.append({
            "query_position": pos,
            "n_unique_focuses": unique_focuses,
            "max_possible": int(model.cfg.n_heads),
        })
    mean_unique = sum(a["n_unique_focuses"] for a in agreements) / len(agreements)
    return {
        "layer": layer,
        "per_query": agreements,
        "mean_unique_focuses": mean_unique,
        "focus_diversity": mean_unique / model.cfg.n_heads,
    }


def head_diversity_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of head diversity."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        pd = pattern_diversity(model, tokens, layer)
        ed = entropy_diversity(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "pattern_similarity": pd["mean_pattern_similarity"],
            "is_pattern_diverse": pd["is_diverse"],
            "entropy_range": ed["entropy_range"],
            "is_entropy_diverse": ed["is_entropy_diverse"],
        })
    return {"per_layer": per_layer}
