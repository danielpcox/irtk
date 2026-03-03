"""Cross-layer alignment: representation alignment between layers."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def adjacent_layer_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                                position: int = -1) -> dict:
    """Cosine alignment between adjacent layers' residual streams.

    High alignment = layer barely changes direction; low = redirects.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    per_pair = []
    for layer in range(model.cfg.n_layers - 1):
        r1 = cache[("resid_post", layer)][position]
        r2 = cache[("resid_post", layer + 1)][position]
        n1 = jnp.sqrt(jnp.sum(r1 ** 2)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(r2 ** 2)).clip(1e-8)
        cos = float(jnp.sum(r1 * r2) / (n1 * n2))
        per_pair.append({
            "layers": (layer, layer + 1),
            "cosine": cos,
            "is_aligned": cos > 0.9,
        })

    cosines = [p["cosine"] for p in per_pair]
    return {
        "position": position,
        "per_pair": per_pair,
        "mean_alignment": sum(cosines) / max(len(cosines), 1),
    }


def full_layer_alignment_matrix(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Full cosine alignment matrix between all pairs of layers.

    Reveals block structure and long-range alignment patterns.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    n_layers = model.cfg.n_layers
    resids = []
    for layer in range(n_layers):
        r = cache[("resid_post", layer)][position]
        norm = jnp.sqrt(jnp.sum(r ** 2)).clip(1e-8)
        resids.append(r / norm)

    matrix = []
    for i in range(n_layers):
        row = []
        for j in range(n_layers):
            cos = float(jnp.sum(resids[i] * resids[j]))
            row.append(cos)
        matrix.append(row)

    return {
        "position": position,
        "alignment_matrix": matrix,
        "n_layers": n_layers,
    }


def component_alignment_across_layers(model: HookedTransformer, tokens: jnp.ndarray,
                                         position: int = -1) -> dict:
    """Alignment of attention and MLP outputs across layers.

    Do later layers' outputs point in similar directions to earlier ones?
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    attn_outs = []
    mlp_outs = []
    for layer in range(model.cfg.n_layers):
        attn = cache[("attn_out", layer)][position]
        mlp = cache[("mlp_out", layer)][position]
        attn_outs.append(attn)
        mlp_outs.append(mlp)

    attn_pairs = []
    for i in range(len(attn_outs) - 1):
        n1 = jnp.sqrt(jnp.sum(attn_outs[i] ** 2)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(attn_outs[i + 1] ** 2)).clip(1e-8)
        cos = float(jnp.sum(attn_outs[i] * attn_outs[i + 1]) / (n1 * n2))
        attn_pairs.append({"layers": (i, i + 1), "cosine": cos})

    mlp_pairs = []
    for i in range(len(mlp_outs) - 1):
        n1 = jnp.sqrt(jnp.sum(mlp_outs[i] ** 2)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(mlp_outs[i + 1] ** 2)).clip(1e-8)
        cos = float(jnp.sum(mlp_outs[i] * mlp_outs[i + 1]) / (n1 * n2))
        mlp_pairs.append({"layers": (i, i + 1), "cosine": cos})

    return {
        "position": position,
        "attn_alignment": attn_pairs,
        "mlp_alignment": mlp_pairs,
    }


def early_late_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """Alignment between early and late layer representations.

    Shows how much the early representation influences the final output.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    n_layers = model.cfg.n_layers
    first = cache[("resid_post", 0)][position]
    last = cache[("resid_post", n_layers - 1)][position]

    f_norm = jnp.sqrt(jnp.sum(first ** 2)).clip(1e-8)
    l_norm = jnp.sqrt(jnp.sum(last ** 2)).clip(1e-8)
    cos = float(jnp.sum(first * last) / (f_norm * l_norm))

    return {
        "position": position,
        "early_late_cosine": cos,
        "early_norm": float(f_norm),
        "late_norm": float(l_norm),
        "norm_growth": float(l_norm / f_norm),
        "is_preserved": cos > 0.5,
    }


def cross_layer_alignment_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Combined cross-layer alignment summary."""
    adj = adjacent_layer_alignment(model, tokens, position)
    el = early_late_alignment(model, tokens, position)
    return {
        "position": adj["position"],
        "mean_adjacent_alignment": adj["mean_alignment"],
        "early_late_cosine": el["early_late_cosine"],
        "norm_growth": el["norm_growth"],
        "is_preserved": el["is_preserved"],
    }
