"""Attention head clustering: group heads by behavioral similarity.

Cluster attention heads based on pattern similarity, output direction
similarity, and functional role to identify head families.
"""

import jax
import jax.numpy as jnp


def head_pattern_similarity(model, tokens, layer=0):
    """Cosine similarity matrix between attention heads based on their patterns.

    Returns:
        dict with 'similarity_matrix' [n_heads, n_heads], 'per_pair' list,
        'mean_similarity' float.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    flat = patterns.reshape(n_heads, -1)
    norms = jnp.linalg.norm(flat, axis=-1, keepdims=True) + 1e-10
    normed = flat / norms
    sim = normed @ normed.T
    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                "head_a": int(i),
                "head_b": int(j),
                "similarity": float(sim[i, j]),
            })
    mask = jnp.ones((n_heads, n_heads)) - jnp.eye(n_heads)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask))
    return {
        "similarity_matrix": sim,
        "per_pair": pairs,
        "mean_similarity": mean_sim,
    }


def head_output_direction_clustering(model, tokens, layer=0):
    """Cluster heads by their output direction similarity.

    Uses cosine similarity of mean head outputs (z @ W_O).

    Returns:
        dict with 'similarity_matrix', 'per_pair', 'mean_similarity'.
    """
    _, cache = model.run_with_cache(tokens)
    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    outputs = jnp.einsum("snh,nhm->nsm", z, W_O)  # [n_heads, seq, d_model]
    mean_out = jnp.mean(outputs, axis=1)  # [n_heads, d_model]
    norms = jnp.linalg.norm(mean_out, axis=-1, keepdims=True) + 1e-10
    normed = mean_out / norms
    sim = normed @ normed.T
    n_heads = sim.shape[0]
    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                "head_a": int(i),
                "head_b": int(j),
                "similarity": float(sim[i, j]),
            })
    mask = jnp.ones((n_heads, n_heads)) - jnp.eye(n_heads)
    mean_sim = float(jnp.sum(sim * mask) / jnp.sum(mask))
    return {
        "similarity_matrix": sim,
        "per_pair": pairs,
        "mean_similarity": mean_sim,
    }


def head_functional_fingerprint(model, tokens, layer=0):
    """Fingerprint each head by key behavioral metrics.

    Computes per-head: entropy, max attention weight, diagonal score,
    mean value norm.

    Returns:
        dict with 'per_head' list of fingerprint dicts.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    z = cache[("z", layer)]  # [n_heads, seq, d_head]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        entropy = float(-jnp.sum(pat * jnp.log(pat + 1e-10)) / seq_len)
        max_weight = float(jnp.max(pat))
        diag_score = float(jnp.mean(jnp.diag(pat[:min(seq_len, pat.shape[1]), :min(seq_len, pat.shape[1])])))
        mean_val_norm = float(jnp.mean(jnp.linalg.norm(z[h], axis=-1)))
        per_head.append({
            "head": int(h),
            "entropy": entropy,
            "max_weight": max_weight,
            "diagonal_score": diag_score,
            "mean_value_norm": mean_val_norm,
        })
    return {"per_head": per_head}


def head_redundancy_score(model, tokens, layer=0):
    """How redundant is each head with the other heads?

    High redundancy = head's pattern is very similar to at least one other head.

    Returns:
        dict with 'per_head' list with max_similarity and is_redundant.
    """
    sim_result = head_pattern_similarity(model, tokens, layer=layer)
    sim_matrix = sim_result["similarity_matrix"]
    n_heads = sim_matrix.shape[0]
    per_head = []
    for h in range(n_heads):
        others = [float(sim_matrix[h, j]) for j in range(n_heads) if j != h]
        max_sim = max(others) if others else 0.0
        most_similar = int(jnp.argmax(
            jnp.where(jnp.arange(n_heads) == h, -jnp.inf, sim_matrix[h])
        ))
        per_head.append({
            "head": int(h),
            "max_similarity": max_sim,
            "most_similar_to": most_similar,
            "is_redundant": max_sim > 0.9,
        })
    n_redundant = sum(1 for p in per_head if p["is_redundant"])
    return {"per_head": per_head, "n_redundant": n_redundant}


def attention_head_clustering_summary(model, tokens):
    """Summary of head clustering across all layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        pat_sim = head_pattern_similarity(model, tokens, layer=layer)
        red = head_redundancy_score(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "mean_pattern_similarity": pat_sim["mean_similarity"],
            "n_redundant": red["n_redundant"],
        })
    return {"per_layer": per_layer}
