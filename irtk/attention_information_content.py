"""Attention information content: information-theoretic analysis of attention."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def attention_entropy_profile(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Entropy of attention patterns for each head and query position.

    High entropy = diffuse attention; low entropy = focused attention.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]

    per_head = []
    for head in range(n_heads):
        p = patterns[head]  # [seq, seq]
        ent = -jnp.sum(p * jnp.log(p.clip(1e-10)), axis=-1)  # [seq]
        max_ent = jnp.log(jnp.arange(1, seq_len + 1, dtype=jnp.float32))
        norm_ent = ent / max_ent.clip(1e-8)
        per_head.append({
            "head": int(head),
            "mean_entropy": float(jnp.mean(ent)),
            "mean_normalized_entropy": float(jnp.mean(norm_ent)),
            "is_sharp": float(jnp.mean(norm_ent)) < 0.3,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_entropy": sum(h["mean_entropy"] for h in per_head) / len(per_head),
    }


def attention_mutual_information(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0) -> dict:
    """Estimate mutual information between head attention and input.

    Higher values mean the head's attention is more input-dependent
    (less uniform/fixed).
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = patterns.shape[1]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = patterns[head]  # [seq, seq]
        # Marginal distribution over keys
        marginal = jnp.mean(p, axis=0)  # [seq] - average attention per key
        marginal_ent = float(-jnp.sum(marginal * jnp.log(marginal.clip(1e-10))))
        # Mean conditional entropy
        cond_ent = float(jnp.mean(-jnp.sum(p * jnp.log(p.clip(1e-10)), axis=-1)))
        mi = marginal_ent - cond_ent
        per_head.append({
            "head": int(head),
            "marginal_entropy": marginal_ent,
            "conditional_entropy": cond_ent,
            "mutual_information": max(mi, 0),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_mi": sum(h["mutual_information"] for h in per_head) / len(per_head),
    }


def attention_concentration(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int = 0, head: int = 0) -> dict:
    """Measure attention concentration: how much mass is in the top-k positions.

    Shows the effective "attention span" of each query.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    p = patterns[head]  # [seq, seq]
    seq_len = p.shape[0]

    per_query = []
    for q in range(seq_len):
        row = p[q]
        sorted_row = jnp.sort(row)[::-1]
        top1 = float(sorted_row[0])
        top3 = float(jnp.sum(sorted_row[:min(3, len(sorted_row))]))
        per_query.append({
            "query": q,
            "top1_mass": top1,
            "top3_mass": top3,
        })
    mean_top1 = sum(pq["top1_mass"] for pq in per_query) / len(per_query)
    return {
        "layer": layer,
        "head": head,
        "per_query": per_query,
        "mean_top1_mass": mean_top1,
        "is_concentrated": mean_top1 > 0.5,
    }


def information_flow_rate(model: HookedTransformer, tokens: jnp.ndarray,
                           layer: int = 0) -> dict:
    """Estimate information flow through attention based on value norms.

    Attention × value norm gives an estimate of information transferred.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    v = cache[("v", layer)]  # [seq, n_heads, d_head]

    per_head = []
    for head in range(model.cfg.n_heads):
        vh = v[:, head, :]  # [seq, d_head]
        v_norms = jnp.sqrt(jnp.sum(vh ** 2, axis=-1))  # [seq]
        # Weighted average value norm per query
        weighted_norms = patterns[head] @ v_norms  # [seq]
        mean_flow = float(jnp.mean(weighted_norms))
        per_head.append({
            "head": int(head),
            "mean_information_flow": mean_flow,
            "max_information_flow": float(jnp.max(weighted_norms)),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_flow": sum(h["mean_information_flow"] for h in per_head) / len(per_head),
    }


def information_content_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer information content summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        ent = attention_entropy_profile(model, tokens, layer)
        mi = attention_mutual_information(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_entropy": ent["mean_entropy"],
            "mean_mi": mi["mean_mi"],
        })
    return {"per_layer": per_layer}
