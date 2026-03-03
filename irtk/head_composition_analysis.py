"""Head composition analysis: how attention heads compose across layers.

Analyze QK and OV composition between heads in different layers,
identifying which early heads feed information to later heads.
"""

import jax.numpy as jnp


def qk_composition_scores(model, source_layer=0, dest_layer=1):
    """QK composition: how much do source heads influence dest head queries?

    Measures composition via W_OV @ W_QK path.

    Returns:
        dict with 'scores' matrix [src_heads, dst_heads], 'per_pair' list.
    """
    W_O_src = model.blocks[source_layer].attn.W_O  # [n_heads, d_head, d_model]
    W_V_src = model.blocks[source_layer].attn.W_V  # [n_heads, d_model, d_head]
    W_Q_dst = model.blocks[dest_layer].attn.W_Q  # [n_heads, d_model, d_head]
    W_K_dst = model.blocks[dest_layer].attn.W_K  # [n_heads, d_model, d_head]
    n_src = W_O_src.shape[0]
    n_dst = W_Q_dst.shape[0]
    scores = jnp.zeros((n_src, n_dst))
    per_pair = []
    for s in range(n_src):
        OV = W_V_src[s] @ W_O_src[s]  # [d_model, d_model]
        for d in range(n_dst):
            QK = W_Q_dst[d] @ W_K_dst[d].T  # [d_head, d_head] -> need d_model space
            comp = jnp.linalg.norm(OV @ W_Q_dst[d])  # composition strength
            scores = scores.at[s, d].set(comp)
            per_pair.append({
                "source_head": int(s),
                "dest_head": int(d),
                "composition_score": float(comp),
            })
    return {"scores": scores, "per_pair": per_pair}


def ov_composition_scores(model, source_layer=0, dest_layer=1):
    """OV composition: how much does source OV output align with dest OV input?

    Returns:
        dict with 'scores' matrix [src_heads, dst_heads], 'per_pair' list.
    """
    W_O_src = model.blocks[source_layer].attn.W_O  # [n_heads, d_head, d_model]
    W_V_src = model.blocks[source_layer].attn.W_V  # [n_heads, d_model, d_head]
    W_V_dst = model.blocks[dest_layer].attn.W_V  # [n_heads, d_model, d_head]
    W_O_dst = model.blocks[dest_layer].attn.W_O  # [n_heads, d_head, d_model]
    n_src = W_O_src.shape[0]
    n_dst = W_V_dst.shape[0]
    scores = jnp.zeros((n_src, n_dst))
    per_pair = []
    for s in range(n_src):
        OV_src = W_V_src[s] @ W_O_src[s]  # [d_model, d_model]
        for d in range(n_dst):
            OV_dst = W_V_dst[d] @ W_O_dst[d]  # [d_model, d_model]
            comp = float(jnp.linalg.norm(OV_src @ OV_dst))
            scores = scores.at[s, d].set(comp)
            per_pair.append({
                "source_head": int(s),
                "dest_head": int(d),
                "composition_score": float(comp),
            })
    return {"scores": scores, "per_pair": per_pair}


def strongest_compositions(model, source_layer=0, dest_layer=1, top_k=5):
    """Find the strongest head-to-head compositions.

    Returns:
        dict with 'qk_top' and 'ov_top' lists of strongest pairs.
    """
    qk = qk_composition_scores(model, source_layer, dest_layer)
    ov = ov_composition_scores(model, source_layer, dest_layer)
    qk_sorted = sorted(qk["per_pair"], key=lambda x: -x["composition_score"])
    ov_sorted = sorted(ov["per_pair"], key=lambda x: -x["composition_score"])
    return {
        "qk_top": qk_sorted[:top_k],
        "ov_top": ov_sorted[:top_k],
    }


def composition_path_strength(model, tokens, source_layer=0, source_head=0, dest_layer=1, dest_head=0):
    """Measure the actual activation-level composition between two heads.

    Returns:
        dict with 'source_output_norm', 'dest_input_norm', 'composition_strength'.
    """
    _, cache = model.run_with_cache(tokens)
    z_src = cache[("z", source_layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[source_layer].attn.W_O  # [n_heads, d_head, d_model]
    src_out = z_src[:, source_head, :] @ W_O[source_head]  # [seq, d_model]
    resid_pre = cache[("resid_pre", dest_layer)]  # [seq, d_model]
    W_Q = model.blocks[dest_layer].attn.W_Q[dest_head]  # [d_model, d_head]
    q_from_src = src_out @ W_Q  # [seq, d_head]
    q_total = resid_pre @ W_Q  # [seq, d_head]
    src_norm = float(jnp.linalg.norm(q_from_src))
    total_norm = float(jnp.linalg.norm(q_total))
    return {
        "source_output_norm": float(jnp.linalg.norm(src_out)),
        "dest_query_norm": total_norm,
        "composition_strength": src_norm / (total_norm + 1e-10),
    }


def head_composition_summary(model, tokens):
    """Summary of head composition across adjacent layer pairs.

    Returns:
        dict with 'per_layer_pair' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_pair = []
    for l in range(n_layers - 1):
        top = strongest_compositions(model, source_layer=l, dest_layer=l + 1, top_k=1)
        qk_best = top["qk_top"][0] if top["qk_top"] else None
        ov_best = top["ov_top"][0] if top["ov_top"] else None
        per_pair.append({
            "source_layer": l,
            "dest_layer": l + 1,
            "strongest_qk": qk_best,
            "strongest_ov": ov_best,
        })
    return {"per_layer_pair": per_pair}
