"""Attention query analysis: properties of query vectors across heads."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def query_norm_profile(model: HookedTransformer, tokens: jnp.ndarray,
                        layer: int = 0) -> dict:
    """Query vector norms across positions and heads.

    Large query norms produce sharper attention; small norms produce diffuse attention.
    """
    _, cache = model.run_with_cache(tokens)
    q = cache[("q", layer)]  # [seq, n_heads, d_head]

    per_head = []
    for head in range(model.cfg.n_heads):
        norms = jnp.sqrt(jnp.sum(q[:, head, :] ** 2, axis=-1))  # [seq]
        per_head.append({
            "head": int(head),
            "mean_norm": float(jnp.mean(norms)),
            "max_norm": float(jnp.max(norms)),
            "std_norm": float(jnp.std(norms)),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_query_norm": sum(h["mean_norm"] for h in per_head) / len(per_head),
    }


def query_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                     layer: int = 0) -> dict:
    """Diversity of query vectors across positions within each head.

    Diverse queries mean the head adapts to each position's needs;
    uniform queries mean position-independent matching.
    """
    _, cache = model.run_with_cache(tokens)
    q = cache[("q", layer)]  # [seq, n_heads, d_head]
    seq_len = q.shape[0]

    per_head = []
    for head in range(model.cfg.n_heads):
        qh = q[:, head, :]  # [seq, d_head]
        norms = jnp.sqrt(jnp.sum(qh ** 2, axis=-1, keepdims=True)).clip(1e-8)
        normed = qh / norms
        sim = normed @ normed.T
        mask = 1.0 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))
        per_head.append({
            "head": int(head),
            "mean_query_similarity": mean_sim,
            "is_diverse": mean_sim < 0.5,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_diversity": sum(1 for h in per_head if h["is_diverse"]) / len(per_head),
    }


def query_key_matching(model: HookedTransformer, tokens: jnp.ndarray,
                        layer: int = 0, head: int = 0) -> dict:
    """Analyze how queries match with keys (pre-softmax scores).

    Shows the score distribution that drives attention.
    """
    _, cache = model.run_with_cache(tokens)
    attn_scores = cache[("attn_scores", layer)]  # [n_heads, seq, seq]
    scores = attn_scores[head]  # [seq, seq]

    mean_score = float(jnp.mean(scores))
    std_score = float(jnp.std(scores))
    max_score = float(jnp.max(scores))
    min_score = float(jnp.min(scores))

    return {
        "layer": layer,
        "head": head,
        "mean_score": mean_score,
        "std_score": std_score,
        "max_score": max_score,
        "min_score": min_score,
        "score_range": max_score - min_score,
    }


def query_subspace_analysis(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0, head: int = 0) -> dict:
    """Analyze the subspace spanned by query vectors."""
    _, cache = model.run_with_cache(tokens)
    q = cache[("q", layer)]
    qh = q[:, head, :]  # [seq, d_head]

    svs = jnp.linalg.svd(qh, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    return {
        "layer": layer,
        "head": head,
        "effective_rank": eff_rank,
        "top_sv": float(svs[0]),
        "sv_ratio": float(svs[0] / svs[-1].clip(1e-8)),
    }


def query_analysis_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer query analysis summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        norms = query_norm_profile(model, tokens, layer)
        div = query_diversity(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_query_norm": norms["mean_query_norm"],
            "diversity_fraction": div["mean_diversity"],
        })
    return {"per_layer": per_layer}
