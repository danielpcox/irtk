"""Weight tying analysis for mechanistic interpretability.

Analyze the relationship between embedding and unembedding matrices:
alignment measurement, effects of tying, embedding-unembed subspace
analysis, and frequency-dependent effects.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def embedding_unembed_alignment(
    model,
    top_k: int = 10,
) -> dict:
    """Measure alignment between embedding and unembedding matrices.

    If W_E and W_U are tied (or approximately tied), their rows/columns
    should be highly aligned.

    Args:
        model: HookedTransformer model.
        top_k: Top aligned/misaligned tokens.

    Returns:
        Dict with mean_cosine_similarity, per_token_alignment,
        most_aligned, least_aligned, is_approximately_tied.
    """
    W_E = np.array(model.embed.W_E)    # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    d_vocab = W_E.shape[0]
    min_vocab = min(d_vocab, W_U.shape[1])

    # Compare each token's embedding with its unembedding column
    cosines = np.zeros(min_vocab)
    for i in range(min_vocab):
        e = W_E[i]
        u = W_U[:, i]
        e_norm = np.linalg.norm(e)
        u_norm = np.linalg.norm(u)
        if e_norm > 1e-10 and u_norm > 1e-10:
            cosines[i] = np.dot(e, u) / (e_norm * u_norm)

    mean_cos = float(np.mean(cosines))

    ranked = np.argsort(cosines)[::-1]
    most_aligned = [(int(i), float(cosines[i])) for i in ranked[:top_k]]
    least_aligned = [(int(i), float(cosines[i])) for i in ranked[-top_k:]]

    is_tied = mean_cos > 0.95

    return {
        "mean_cosine_similarity": mean_cos,
        "per_token_alignment": jnp.array(cosines),
        "most_aligned": most_aligned,
        "least_aligned": least_aligned,
        "is_approximately_tied": is_tied,
    }


def embedding_subspace_analysis(
    model,
    n_components: int = 10,
) -> dict:
    """Analyze the subspace structure of embedding and unembedding.

    Compares the principal subspaces of W_E and W_U to understand
    how they share (or don't share) structure.

    Args:
        model: HookedTransformer model.
        n_components: Number of components to compare.

    Returns:
        Dict with embedding_rank, unembed_rank, subspace_overlap,
        shared_dimensions.
    """
    W_E = np.array(model.embed.W_E)    # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # SVD of each
    U_E, S_E, Vt_E = np.linalg.svd(W_E, full_matrices=False)
    U_U, S_U, Vt_U = np.linalg.svd(W_U, full_matrices=False)

    k = min(n_components, len(S_E), len(S_U))

    # Effective rank
    def effective_rank(S):
        p = S / (np.sum(S) + 1e-10)
        return float(np.exp(-np.sum(p * np.log(p + 1e-10))))

    e_rank = effective_rank(S_E)
    u_rank = effective_rank(S_U)

    # Subspace overlap: cosine between top-k right singular vectors
    # W_E's row space (Vt_E) vs W_U's column space (U_U)
    V_E = Vt_E[:k].T  # [d_model, k]
    V_U = U_U[:, :k]   # [d_model, k]

    overlap_matrix = V_E.T @ V_U  # [k, k]
    overlap_score = float(np.linalg.norm(overlap_matrix, 'fro') / k)

    # Shared dimensions: singular values of overlap
    overlap_svs = np.linalg.svd(overlap_matrix, compute_uv=False)
    shared = int(np.sum(overlap_svs > 0.5))

    return {
        "embedding_effective_rank": e_rank,
        "unembed_effective_rank": u_rank,
        "subspace_overlap": overlap_score,
        "shared_dimensions": shared,
        "overlap_singular_values": overlap_svs[:k].tolist(),
        "embedding_top_singular_values": S_E[:k].tolist(),
        "unembed_top_singular_values": S_U[:k].tolist(),
    }


def norm_distribution(
    model,
    top_k: int = 10,
) -> dict:
    """Analyze the norm distribution of embedding and unembedding vectors.

    Args:
        model: HookedTransformer model.
        top_k: Top tokens by norm.

    Returns:
        Dict with embedding_norms, unembed_norms, norm_correlation,
        highest/lowest norm tokens.
    """
    W_E = np.array(model.embed.W_E)    # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    e_norms = np.linalg.norm(W_E, axis=-1)  # [d_vocab]
    u_norms = np.linalg.norm(W_U, axis=0)   # [d_vocab]

    min_vocab = min(len(e_norms), len(u_norms))
    if min_vocab > 1:
        corr = float(np.corrcoef(e_norms[:min_vocab], u_norms[:min_vocab])[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    e_ranked = np.argsort(e_norms)[::-1]
    highest_embed = [(int(i), float(e_norms[i])) for i in e_ranked[:top_k]]
    lowest_embed = [(int(i), float(e_norms[i])) for i in e_ranked[-top_k:]]

    return {
        "embedding_norms": jnp.array(e_norms),
        "unembed_norms": jnp.array(u_norms),
        "norm_correlation": corr,
        "highest_embed_norm": highest_embed,
        "lowest_embed_norm": lowest_embed,
        "mean_embed_norm": float(np.mean(e_norms)),
        "mean_unembed_norm": float(np.mean(u_norms)),
    }


def embedding_isotropy(
    model,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Measure isotropy of the embedding space.

    Isotropic embeddings have uniform directional coverage.
    Anisotropic embeddings cluster in certain directions.

    Args:
        model: HookedTransformer model.
        n_samples: Pairs to sample for estimating isotropy.
        seed: Random seed.

    Returns:
        Dict with isotropy_score, mean_cosine, std_cosine,
        principal_direction_dominance.
    """
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    d_vocab = W_E.shape[0]

    rng = np.random.RandomState(seed)
    n_samples = min(n_samples, d_vocab * (d_vocab - 1) // 2)

    # Sample random pairs and compute cosines
    cosines = []
    for _ in range(n_samples):
        i, j = rng.choice(d_vocab, 2, replace=False)
        e_i = W_E[i]
        e_j = W_E[j]
        n_i = np.linalg.norm(e_i)
        n_j = np.linalg.norm(e_j)
        if n_i > 1e-10 and n_j > 1e-10:
            cosines.append(float(np.dot(e_i, e_j) / (n_i * n_j)))

    cosines = np.array(cosines)
    mean_cos = float(np.mean(cosines))
    std_cos = float(np.std(cosines))

    # Isotropy: 1 - |mean_cosine| (perfect isotropy = mean_cosine close to 0)
    isotropy = 1.0 - abs(mean_cos)

    # Principal direction dominance: how much variance is in top-1 direction
    U, S, Vt = np.linalg.svd(W_E, full_matrices=False)
    dominance = float(S[0] ** 2 / (np.sum(S ** 2) + 1e-10))

    return {
        "isotropy_score": isotropy,
        "mean_cosine": mean_cos,
        "std_cosine": std_cos,
        "principal_direction_dominance": dominance,
        "n_samples": len(cosines),
    }


def token_neighborhood_analysis(
    model,
    token_ids: Optional[list] = None,
    k: int = 5,
) -> dict:
    """Analyze local neighborhoods in embedding space.

    For each token, finds nearest neighbors in both embedding and
    unembedding space, and checks consistency.

    Args:
        model: HookedTransformer model.
        token_ids: Tokens to analyze (default: first 10).
        k: Number of neighbors.

    Returns:
        Dict with per_token neighbors in embed/unembed space,
        neighborhood_consistency.
    """
    W_E = np.array(model.embed.W_E)
    W_U = np.array(model.unembed.W_U)

    if token_ids is None:
        token_ids = list(range(min(10, W_E.shape[0])))

    # Normalize
    E_normed = W_E / (np.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10)
    U_normed = W_U / (np.linalg.norm(W_U, axis=0, keepdims=True) + 1e-10)

    per_token = []
    consistencies = []

    for tid in token_ids:
        if tid >= W_E.shape[0]:
            continue

        # Embed neighbors
        e_sims = E_normed @ E_normed[tid]
        e_sims[tid] = -2  # exclude self
        e_neighbors = np.argsort(e_sims)[::-1][:k]

        # Unembed neighbors
        u_sims = U_normed.T @ U_normed[:, tid]
        u_sims[tid] = -2
        u_neighbors = np.argsort(u_sims)[::-1][:k]

        # Consistency: overlap between embed and unembed neighbors
        overlap = len(set(e_neighbors.tolist()) & set(u_neighbors.tolist()))
        consistency = overlap / k
        consistencies.append(consistency)

        per_token.append({
            "token_id": tid,
            "embed_neighbors": e_neighbors.tolist(),
            "unembed_neighbors": u_neighbors.tolist(),
            "neighborhood_consistency": consistency,
        })

    return {
        "per_token": per_token,
        "mean_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
    }
