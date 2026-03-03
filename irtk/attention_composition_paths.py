"""Attention composition paths: multi-hop attention and composition chains."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def two_hop_attention(model: HookedTransformer, tokens: jnp.ndarray,
                        layer1: int = 0, layer2: int = 1) -> dict:
    """Compose attention patterns from two layers for multi-hop paths.

    Virtual attention = layer2_pattern @ layer1_pattern.
    """
    _, cache = model.run_with_cache(tokens)
    p1 = cache[("pattern", layer1)]  # [n_heads, seq, seq]
    p2 = cache[("pattern", layer2)]  # [n_heads, seq, seq]

    per_head_pair = []
    for h1 in range(model.cfg.n_heads):
        for h2 in range(model.cfg.n_heads):
            virtual = p2[h2] @ p1[h1]  # [seq, seq]
            # How different is virtual from direct?
            direct = p2[h2]
            v_flat = virtual.flatten()
            d_flat = direct.flatten()
            v_norm = jnp.sqrt(jnp.sum(v_flat ** 2)).clip(1e-8)
            d_norm = jnp.sqrt(jnp.sum(d_flat ** 2)).clip(1e-8)
            cos = float(jnp.sum(v_flat * d_flat) / (v_norm * d_norm))

            per_head_pair.append({
                "head1": int(h1),
                "head2": int(h2),
                "virtual_direct_similarity": cos,
            })

    return {
        "layer1": layer1,
        "layer2": layer2,
        "per_head_pair": per_head_pair,
        "n_pairs": len(per_head_pair),
    }


def attention_path_strength(model: HookedTransformer, tokens: jnp.ndarray,
                               source: int = 0, target: int = -1) -> dict:
    """Strength of attention paths from source to target across layers.

    Shows how information flows from source to target through attention.
    """
    _, cache = model.run_with_cache(tokens)
    if target < 0:
        target = len(tokens) + target

    per_layer = []
    for layer in range(model.cfg.n_layers):
        pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]

        per_head = []
        for head in range(model.cfg.n_heads):
            strength = float(pattern[head, target, source])
            per_head.append({
                "head": int(head),
                "attention_to_source": strength,
            })

        max_strength = max(h["attention_to_source"] for h in per_head)
        per_layer.append({
            "layer": layer,
            "per_head": per_head,
            "max_strength": max_strength,
            "strongest_head": max(range(len(per_head)),
                                   key=lambda i: per_head[i]["attention_to_source"]),
        })
    return {
        "source": source,
        "target": target,
        "per_layer": per_layer,
    }


def composition_score_matrix(model: HookedTransformer, tokens: jnp.ndarray,
                                layer1: int = 0, layer2: int = 1) -> dict:
    """OV-QK composition scores between heads in two layers.

    Measures how much layer1's output aligns with layer2's query space.
    """
    _, cache = model.run_with_cache(tokens)

    # Get head outputs from layer1 and queries from layer2
    z1 = cache[("z", layer1)]  # [seq, n_heads, d_head]
    q2 = cache[("q", layer2)]  # [seq, n_heads, d_head]

    per_pair = []
    for h1 in range(model.cfg.n_heads):
        for h2 in range(model.cfg.n_heads):
            # Mean alignment between layer1 head h1 output and layer2 head h2 queries
            z = z1[:, h1, :]  # [seq, d_head]
            q = q2[:, h2, :]  # [seq, d_head]
            z_norm = jnp.sqrt(jnp.sum(z ** 2, axis=-1)).clip(1e-8)
            q_norm = jnp.sqrt(jnp.sum(q ** 2, axis=-1)).clip(1e-8)
            cos = jnp.sum(z * q, axis=-1) / (z_norm * q_norm)
            mean_cos = float(jnp.mean(cos))

            per_pair.append({
                "head1": int(h1),
                "head2": int(h2),
                "composition_score": mean_cos,
            })

    scores = [p["composition_score"] for p in per_pair]
    return {
        "layer1": layer1,
        "layer2": layer2,
        "per_pair": per_pair,
        "max_score": max(scores),
        "mean_score": sum(scores) / len(scores),
    }


def attention_chain_strength(model: HookedTransformer, tokens: jnp.ndarray,
                                source: int = 0, target: int = -1) -> dict:
    """Cumulative attention path strength through all layers.

    Product of per-layer attention from source to target.
    """
    _, cache = model.run_with_cache(tokens)
    if target < 0:
        target = len(tokens) + target

    # Track per-head maximum path to target from source
    per_head_per_layer = []
    for layer in range(model.cfg.n_layers):
        pattern = cache[("pattern", layer)]
        heads = []
        for head in range(model.cfg.n_heads):
            heads.append(float(pattern[head, target, source]))
        per_head_per_layer.append(heads)

    # Strongest path: max over heads at each layer, product
    max_per_layer = [max(heads) for heads in per_head_per_layer]
    chain_strength = 1.0
    for s in max_per_layer:
        chain_strength *= s

    return {
        "source": source,
        "target": target,
        "max_per_layer": max_per_layer,
        "chain_strength": chain_strength,
        "is_strong_path": chain_strength > 0.01,
    }


def attention_composition_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer attention composition summary."""
    per_layer_pair = []
    for l1 in range(model.cfg.n_layers - 1):
        l2 = l1 + 1
        comp = composition_score_matrix(model, tokens, l1, l2)
        per_layer_pair.append({
            "layers": (l1, l2),
            "max_composition": comp["max_score"],
            "mean_composition": comp["mean_score"],
        })
    return {"per_layer_pair": per_layer_pair}
