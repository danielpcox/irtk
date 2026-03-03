"""Embedding composition analysis: how token and positional embeddings compose."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def token_position_balance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Compare norms of token and positional embeddings.

    Shows the relative contribution of token identity vs position.
    """
    _, cache = model.run_with_cache(tokens)
    tok_embed = cache["hook_embed"]  # [seq, d_model]
    pos_embed = cache["hook_pos_embed"]  # [seq, d_model]

    per_position = []
    for pos in range(len(tokens)):
        t_norm = float(jnp.sqrt(jnp.sum(tok_embed[pos] ** 2)))
        p_norm = float(jnp.sqrt(jnp.sum(pos_embed[pos] ** 2)))
        total = t_norm + p_norm
        per_position.append({
            "position": pos,
            "token_norm": t_norm,
            "position_norm": p_norm,
            "token_fraction": t_norm / max(total, 1e-8),
        })
    mean_tok_frac = sum(p["token_fraction"] for p in per_position) / len(per_position)
    return {
        "per_position": per_position,
        "mean_token_fraction": mean_tok_frac,
        "dominant": "token" if mean_tok_frac > 0.5 else "position",
    }


def token_position_alignment(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cosine similarity between token and positional embeddings.

    High alignment means position and identity point in similar directions;
    low alignment means they provide independent information.
    """
    _, cache = model.run_with_cache(tokens)
    tok_embed = cache["hook_embed"]
    pos_embed = cache["hook_pos_embed"]

    per_position = []
    for pos in range(len(tokens)):
        t = tok_embed[pos]
        p = pos_embed[pos]
        t_norm = jnp.sqrt(jnp.sum(t ** 2)).clip(1e-8)
        p_norm = jnp.sqrt(jnp.sum(p ** 2)).clip(1e-8)
        cos = float(jnp.sum(t * p) / (t_norm * p_norm))
        per_position.append({
            "position": pos,
            "cosine": cos,
        })
    cosines = [p["cosine"] for p in per_position]
    return {
        "per_position": per_position,
        "mean_alignment": sum(cosines) / len(cosines),
        "is_aligned": abs(sum(cosines) / len(cosines)) > 0.3,
    }


def combined_embedding_properties(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Properties of the combined (token + position) embedding.

    Analyzes the initial residual stream representation.
    """
    _, cache = model.run_with_cache(tokens)
    tok_embed = cache["hook_embed"]
    pos_embed = cache["hook_pos_embed"]
    combined = tok_embed + pos_embed  # [seq, d_model]

    # Pairwise similarity between positions
    norms = jnp.sqrt(jnp.sum(combined ** 2, axis=-1, keepdims=True)).clip(1e-8)
    normed = combined / norms
    sim = normed @ normed.T
    seq_len = combined.shape[0]
    mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask).clip(1e-8))

    per_position = []
    for pos in range(seq_len):
        per_position.append({
            "position": pos,
            "combined_norm": float(norms[pos, 0]),
        })
    return {
        "per_position": per_position,
        "mean_pairwise_similarity": mean_sim,
        "mean_norm": float(jnp.mean(norms)),
    }


def embedding_subspace_analysis(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze the subspace occupied by the initial embeddings.

    Uses SVD to find effective dimensionality.
    """
    _, cache = model.run_with_cache(tokens)
    tok_embed = cache["hook_embed"]
    pos_embed = cache["hook_pos_embed"]
    combined = tok_embed + pos_embed

    svs = jnp.linalg.svd(combined, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    cumvar = jnp.cumsum(svs ** 2) / jnp.sum(svs ** 2).clip(1e-8)
    dim_90 = int(jnp.searchsorted(cumvar, 0.9)) + 1

    return {
        "effective_rank": eff_rank,
        "dim_for_90_pct": dim_90,
        "top_singular_value": float(svs[0]),
        "n_tokens": int(len(tokens)),
    }


def embedding_composition_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Combined embedding composition analysis."""
    balance = token_position_balance(model, tokens)
    align = token_position_alignment(model, tokens)
    props = combined_embedding_properties(model, tokens)
    subspace = embedding_subspace_analysis(model, tokens)
    return {
        "token_fraction": balance["mean_token_fraction"],
        "dominant": balance["dominant"],
        "tp_alignment": align["mean_alignment"],
        "pairwise_similarity": props["mean_pairwise_similarity"],
        "effective_rank": subspace["effective_rank"],
        "mean_norm": props["mean_norm"],
    }
