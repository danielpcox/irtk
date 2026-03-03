"""Embedding space geometry: analyze the structure of token embeddings.

Study token embedding clustering, nearest neighbors, isotropy,
and the relationship between embedding and unembedding spaces.
"""

import jax.numpy as jnp


def embedding_isotropy(model):
    """Measure how isotropic (uniformly distributed) the embeddings are.

    Perfect isotropy = 1.0. Low isotropy means embeddings cluster.

    Returns:
        dict with 'isotropy_score', 'mean_cosine', 'std_cosine'.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    norms = jnp.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10
    normed = W_E / norms
    n = min(normed.shape[0], 200)  # sample for efficiency
    sample = normed[:n]
    sim = sample @ sample.T
    mask = jnp.ones((n, n)) - jnp.eye(n)
    off_diag = sim * mask
    mean_cos = float(jnp.sum(off_diag) / jnp.sum(mask))
    std_cos = float(jnp.sqrt(jnp.sum((off_diag - mean_cos * mask) ** 2) / jnp.sum(mask)))
    isotropy = 1.0 - abs(mean_cos)
    return {
        "isotropy_score": isotropy,
        "mean_cosine": mean_cos,
        "std_cosine": std_cos,
    }


def embedding_nearest_neighbors(model, token_id, top_k=5):
    """Find the nearest neighbor tokens to a given token in embedding space.

    Returns:
        dict with 'query_token', 'neighbors' list of (token, similarity).
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    query = W_E[token_id]
    norms = jnp.linalg.norm(W_E, axis=-1) + 1e-10
    query_norm = jnp.linalg.norm(query) + 1e-10
    sims = (W_E @ query) / (norms * query_norm)
    top_idx = jnp.argsort(-sims)[:top_k + 1]  # +1 to exclude self
    neighbors = []
    for idx in top_idx:
        idx_int = int(idx)
        if idx_int != token_id:
            neighbors.append((idx_int, float(sims[idx])))
        if len(neighbors) == top_k:
            break
    return {
        "query_token": int(token_id),
        "neighbors": neighbors,
    }


def embed_unembed_alignment(model, top_k=5):
    """How well do embedding and unembedding directions align per token?

    If perfectly aligned, the model favors "copying" through the residual.

    Returns:
        dict with 'per_token_cosine' array, 'mean_alignment', 'most_aligned'.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    d_vocab = W_E.shape[0]
    cosines = []
    for t in range(d_vocab):
        e = W_E[t]
        u = W_U[:, t]
        cos = float(jnp.dot(e, u) / (jnp.linalg.norm(e) * jnp.linalg.norm(u) + 1e-10))
        cosines.append(cos)
    cosines_arr = jnp.array(cosines)
    top_idx = jnp.argsort(-cosines_arr)[:top_k]
    most_aligned = [(int(i), float(cosines_arr[i])) for i in top_idx]
    return {
        "per_token_cosine": cosines_arr,
        "mean_alignment": float(jnp.mean(cosines_arr)),
        "most_aligned": most_aligned,
    }


def embedding_effective_dimension(model):
    """Effective dimensionality of the embedding space via SVD.

    Returns:
        dict with 'effective_rank', 'd_model', 'utilization_ratio'.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    s = jnp.linalg.svd(W_E, compute_uv=False)
    s_norm = s / (jnp.sum(s) + 1e-10)
    s_safe = jnp.where(s_norm > 1e-10, s_norm, 1e-10)
    eff_rank = float(jnp.exp(-jnp.sum(s_safe * jnp.log(s_safe))))
    d_model = int(W_E.shape[1])
    return {
        "effective_rank": eff_rank,
        "d_model": d_model,
        "utilization_ratio": eff_rank / d_model,
    }


def embedding_geometry_summary(model):
    """Summary of embedding space geometry.

    Returns:
        dict with isotropy, dimension, and alignment info.
    """
    iso = embedding_isotropy(model)
    dim = embedding_effective_dimension(model)
    align = embed_unembed_alignment(model, top_k=3)
    return {
        "isotropy": iso["isotropy_score"],
        "mean_cosine": iso["mean_cosine"],
        "effective_rank": dim["effective_rank"],
        "utilization": dim["utilization_ratio"],
        "mean_eu_alignment": align["mean_alignment"],
    }
