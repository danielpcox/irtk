"""Attention weight matrix structure: properties of QKV and output weight matrices."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def qk_weight_norms(model: HookedTransformer) -> dict:
    """Compute norms of W_Q and W_K weight matrices per head per layer.

    Shows which heads have strong vs weak query/key projections.
    """
    per_layer = []
    for layer in range(model.cfg.n_layers):
        W_Q = model.blocks[layer].attn.W_Q  # [n_heads, d_model, d_head]
        W_K = model.blocks[layer].attn.W_K
        per_head = []
        for head in range(model.cfg.n_heads):
            q_norm = float(jnp.sqrt(jnp.sum(W_Q[head] ** 2)))
            k_norm = float(jnp.sqrt(jnp.sum(W_K[head] ** 2)))
            per_head.append({
                "head": int(head),
                "q_norm": q_norm,
                "k_norm": k_norm,
                "qk_ratio": q_norm / max(k_norm, 1e-8),
            })
        per_layer.append({
            "layer": layer,
            "per_head": per_head,
            "mean_q_norm": sum(h["q_norm"] for h in per_head) / len(per_head),
            "mean_k_norm": sum(h["k_norm"] for h in per_head) / len(per_head),
        })
    return {"per_layer": per_layer}


def ov_weight_norms(model: HookedTransformer) -> dict:
    """Compute norms of W_V and W_O weight matrices per head per layer.

    Shows which heads have strong value/output projections (large
    potential to modify the residual stream).
    """
    per_layer = []
    for layer in range(model.cfg.n_layers):
        W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
        W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
        per_head = []
        for head in range(model.cfg.n_heads):
            v_norm = float(jnp.sqrt(jnp.sum(W_V[head] ** 2)))
            o_norm = float(jnp.sqrt(jnp.sum(W_O[head] ** 2)))
            per_head.append({
                "head": int(head),
                "v_norm": v_norm,
                "o_norm": o_norm,
                "ov_product_norm": v_norm * o_norm,
            })
        per_layer.append({
            "layer": layer,
            "per_head": per_head,
            "mean_v_norm": sum(h["v_norm"] for h in per_head) / len(per_head),
            "mean_o_norm": sum(h["o_norm"] for h in per_head) / len(per_head),
        })
    return {"per_layer": per_layer}


def weight_matrix_rank_analysis(model: HookedTransformer, layer: int = 0,
                                 head: int = 0) -> dict:
    """Analyze effective rank of QK and OV weight matrices.

    Low rank indicates the head operates in a low-dimensional subspace.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    results = {}
    for name, W in [("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V), ("W_O", W_O)]:
        svs = jnp.linalg.svd(W, compute_uv=False)
        svs_norm = svs / jnp.sum(svs).clip(1e-8)
        eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))
        results[name] = {
            "frobenius_norm": float(jnp.sqrt(jnp.sum(W ** 2))),
            "max_sv": float(svs[0]),
            "min_sv": float(svs[-1]),
            "effective_rank": eff_rank,
            "condition_number": float(svs[0] / svs[-1].clip(1e-8)),
        }
    return {
        "layer": layer,
        "head": head,
        "matrices": results,
    }


def qk_ov_alignment(model: HookedTransformer, layer: int = 0,
                      head: int = 0) -> dict:
    """Measure alignment between QK and OV circuits.

    Computes how much the QK circuit (what to attend to) aligns with
    the OV circuit (what to copy).
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    # QK circuit: W_Q^T @ W_K (d_head, d_head)
    QK = W_Q.T @ W_K  # [d_head, d_head]  (ignoring d_model-space structure)
    # OV circuit: W_V @ W_O (d_model, d_model) projected down
    OV = W_V.T @ W_O.T  # [d_head, d_head] (ignoring outer structure)

    # Flatten and compute cosine
    qk_flat = QK.flatten()
    ov_flat = OV.flatten()
    qk_norm = jnp.sqrt(jnp.sum(qk_flat ** 2)).clip(1e-8)
    ov_norm = jnp.sqrt(jnp.sum(ov_flat ** 2)).clip(1e-8)
    alignment = float(jnp.sum(qk_flat * ov_flat) / (qk_norm * ov_norm))

    return {
        "layer": layer,
        "head": head,
        "qk_ov_alignment": alignment,
        "qk_norm": float(qk_norm),
        "ov_norm": float(ov_norm),
        "is_aligned": abs(alignment) > 0.5,
    }


def weight_structure_summary(model: HookedTransformer) -> dict:
    """Summary of attention weight structure across all heads."""
    qk = qk_weight_norms(model)
    ov = ov_weight_norms(model)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        per_layer.append({
            "layer": layer,
            "mean_q_norm": qk["per_layer"][layer]["mean_q_norm"],
            "mean_k_norm": qk["per_layer"][layer]["mean_k_norm"],
            "mean_v_norm": ov["per_layer"][layer]["mean_v_norm"],
            "mean_o_norm": ov["per_layer"][layer]["mean_o_norm"],
        })
    return {"per_layer": per_layer}
