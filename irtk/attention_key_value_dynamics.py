"""Attention key/value dynamics: how K and V representations evolve across layers."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def key_norm_evolution(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track key vector norms across layers and heads.

    Shows how key magnitudes change through the model, indicating
    which layers create strong vs weak key signals.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        k = cache[("k", layer)]  # [seq, n_heads, d_head]
        norms = jnp.sqrt(jnp.sum(k ** 2, axis=-1))  # [seq, n_heads]
        per_head = []
        for head in range(model.cfg.n_heads):
            head_norms = norms[:, head]
            per_head.append({
                "head": int(head),
                "mean_norm": float(jnp.mean(head_norms)),
                "max_norm": float(jnp.max(head_norms)),
                "min_norm": float(jnp.min(head_norms)),
                "std_norm": float(jnp.std(head_norms)),
            })
        per_layer.append({
            "layer": layer,
            "mean_key_norm": float(jnp.mean(norms)),
            "per_head": per_head,
        })
    norms_all = [p["mean_key_norm"] for p in per_layer]
    return {
        "per_layer": per_layer,
        "norm_trend": "increasing" if norms_all[-1] > norms_all[0] * 1.2 else
                      "decreasing" if norms_all[-1] < norms_all[0] * 0.8 else "stable",
        "max_norm_layer": int(jnp.argmax(jnp.array(norms_all))),
    }


def value_norm_evolution(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track value vector norms across layers and heads.

    Value norms indicate how much information each head writes into
    the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        v = cache[("v", layer)]  # [seq, n_heads, d_head]
        norms = jnp.sqrt(jnp.sum(v ** 2, axis=-1))  # [seq, n_heads]
        per_head = []
        for head in range(model.cfg.n_heads):
            head_norms = norms[:, head]
            per_head.append({
                "head": int(head),
                "mean_norm": float(jnp.mean(head_norms)),
                "max_norm": float(jnp.max(head_norms)),
            })
        per_layer.append({
            "layer": layer,
            "mean_value_norm": float(jnp.mean(norms)),
            "per_head": per_head,
        })
    norms_all = [p["mean_value_norm"] for p in per_layer]
    return {
        "per_layer": per_layer,
        "norm_trend": "increasing" if norms_all[-1] > norms_all[0] * 1.2 else
                      "decreasing" if norms_all[-1] < norms_all[0] * 0.8 else "stable",
        "max_norm_layer": int(jnp.argmax(jnp.array(norms_all))),
    }


def key_value_alignment(model: HookedTransformer, tokens: jnp.ndarray,
                         layer: int = 0) -> dict:
    """Measure alignment between key and value vectors at each position.

    High K-V alignment means a position's key and value point in similar
    directions. Low alignment means the "what to attend to" differs from
    "what information to retrieve".
    """
    _, cache = model.run_with_cache(tokens)
    k = cache[("k", layer)]  # [seq, n_heads, d_head]
    v = cache[("v", layer)]  # [seq, n_heads, d_head]
    seq_len = k.shape[0]

    per_head = []
    for head in range(model.cfg.n_heads):
        kh = k[:, head, :]  # [seq, d_head]
        vh = v[:, head, :]  # [seq, d_head]
        k_norm = jnp.sqrt(jnp.sum(kh ** 2, axis=-1, keepdims=True)).clip(1e-8)
        v_norm = jnp.sqrt(jnp.sum(vh ** 2, axis=-1, keepdims=True)).clip(1e-8)
        cosines = jnp.sum((kh / k_norm) * (vh / v_norm), axis=-1)  # [seq]
        per_head.append({
            "head": int(head),
            "mean_alignment": float(jnp.mean(cosines)),
            "min_alignment": float(jnp.min(cosines)),
            "max_alignment": float(jnp.max(cosines)),
        })
    mean_all = float(jnp.mean(jnp.array([h["mean_alignment"] for h in per_head])))
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_alignment": mean_all,
        "is_aligned": mean_all > 0.5,
    }


def key_similarity_across_positions(model: HookedTransformer, tokens: jnp.ndarray,
                                     layer: int = 0, head: int = 0) -> dict:
    """Pairwise cosine similarity between key vectors at different positions.

    Shows how similar the keys are: uniform keys mean all positions look
    alike to queries, while diverse keys enable selective attention.
    """
    _, cache = model.run_with_cache(tokens)
    k = cache[("k", layer)]  # [seq, n_heads, d_head]
    kh = k[:, head, :]  # [seq, d_head]
    seq_len = kh.shape[0]

    norms = jnp.sqrt(jnp.sum(kh ** 2, axis=-1, keepdims=True)).clip(1e-8)
    kh_normed = kh / norms
    sim_matrix = kh_normed @ kh_normed.T  # [seq, seq]

    # Mean off-diagonal similarity
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim_matrix * mask) / jnp.sum(mask).clip(1e-8))

    return {
        "layer": layer,
        "head": head,
        "mean_key_similarity": mean_sim,
        "is_diverse": mean_sim < 0.5,
        "similarity_matrix_shape": list(sim_matrix.shape),
    }


def kv_dynamics_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of key-value dynamics."""
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        k = cache[("k", layer)]
        v = cache[("v", layer)]
        k_norms = jnp.sqrt(jnp.sum(k ** 2, axis=-1))
        v_norms = jnp.sqrt(jnp.sum(v ** 2, axis=-1))

        # K-V alignment per head averaged
        alignments = []
        for head in range(model.cfg.n_heads):
            kh = k[:, head, :]
            vh = v[:, head, :]
            kn = jnp.sqrt(jnp.sum(kh ** 2, axis=-1, keepdims=True)).clip(1e-8)
            vn = jnp.sqrt(jnp.sum(vh ** 2, axis=-1, keepdims=True)).clip(1e-8)
            cos = jnp.mean(jnp.sum((kh / kn) * (vh / vn), axis=-1))
            alignments.append(float(cos))

        per_layer.append({
            "layer": layer,
            "mean_key_norm": float(jnp.mean(k_norms)),
            "mean_value_norm": float(jnp.mean(v_norms)),
            "mean_kv_alignment": float(jnp.mean(jnp.array(alignments))),
            "kv_norm_ratio": float(jnp.mean(k_norms) / jnp.mean(v_norms).clip(1e-8)),
        })
    return {"per_layer": per_layer}
