"""Embedding space analysis.

Analyze the structure of embedding and unembedding spaces:
neighborhood structure, isotropy, frequency effects, and alignment.
"""

import jax
import jax.numpy as jnp


def embedding_isotropy(model):
    """Measure how isotropic (uniformly distributed) embeddings are.

    Returns:
        dict with isotropy metrics for W_E and W_U.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    def _isotropy(mat):
        # Center
        centered = mat - jnp.mean(mat, axis=0, keepdims=True)
        # SVD
        S = jnp.linalg.svd(centered, compute_uv=False)
        # Isotropy: ratio of min to max singular value (1 = perfectly isotropic)
        ratio = float(S[-1] / jnp.maximum(S[0], 1e-10))
        # Also: effective dimensionality
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_dim = float(jnp.exp(jnp.array(entropy)))
        return {
            'isotropy_ratio': ratio,
            'effective_dimensionality': eff_dim,
            'condition_number': float(S[0] / jnp.maximum(S[-1], 1e-10)),
            'top_singular_value': float(S[0]),
        }

    return {
        'embedding': _isotropy(W_E),
        'unembedding': _isotropy(W_U.T),
    }


def embedding_neighborhood(model, token_ids, k=5):
    """Find nearest neighbors in embedding space.

    Args:
        token_ids: list of token IDs to find neighbors for
        k: number of neighbors

    Returns:
        dict with per_token neighbor lists.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    norms = jnp.linalg.norm(W_E, axis=-1, keepdims=True)
    normalized = W_E / jnp.maximum(norms, 1e-10)

    results = []
    for tid in token_ids:
        tid = int(tid)
        query = normalized[tid]  # [d_model]
        similarities = normalized @ query  # [d_vocab]

        # Top-k (excluding self)
        similarities = similarities.at[tid].set(-1.0)
        top_indices = jnp.argsort(-similarities)[:k]

        neighbors = []
        for idx in top_indices:
            idx = int(idx)
            neighbors.append({
                'token': idx,
                'cosine_similarity': float(similarities[idx]),
            })

        results.append({
            'token': tid,
            'embedding_norm': float(norms[tid, 0]),
            'neighbors': neighbors,
        })

    return {'per_token': results}


def embed_unembed_correspondence(model, top_k=10):
    """How well do embedding and unembedding directions correspond?

    Returns:
        dict with per-token alignment and statistics.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    d_vocab = W_E.shape[0]

    # Per-token cosine between embed and unembed direction
    e_norms = jnp.linalg.norm(W_E, axis=-1)  # [d_vocab]
    u_norms = jnp.linalg.norm(W_U, axis=0)  # [d_vocab]

    alignments = jnp.sum(W_E * W_U.T, axis=-1) / jnp.maximum(e_norms * u_norms, 1e-10)

    # Most and least aligned
    top_aligned = jnp.argsort(-alignments)[:top_k]
    bottom_aligned = jnp.argsort(alignments)[:top_k]

    most_aligned = [{'token': int(t), 'cosine': float(alignments[t])} for t in top_aligned]
    least_aligned = [{'token': int(t), 'cosine': float(alignments[t])} for t in bottom_aligned]

    return {
        'mean_alignment': float(jnp.mean(alignments)),
        'std_alignment': float(jnp.std(alignments)),
        'most_aligned': most_aligned,
        'least_aligned': least_aligned,
    }


def embedding_norm_distribution(model):
    """Analyze the distribution of embedding norms.

    Returns:
        dict with norm statistics.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    norms = jnp.linalg.norm(W_E, axis=-1)  # [d_vocab]

    return {
        'mean_norm': float(jnp.mean(norms)),
        'std_norm': float(jnp.std(norms)),
        'min_norm': float(jnp.min(norms)),
        'max_norm': float(jnp.max(norms)),
        'cv': float(jnp.std(norms) / jnp.maximum(jnp.mean(norms), 1e-10)),
        'n_tokens': int(norms.shape[0]),
    }


def embedding_subspace_structure(model, n_components=5):
    """Analyze the subspace structure of embeddings.

    Returns:
        dict with PCA-like analysis of the embedding matrix.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    centered = W_E - jnp.mean(W_E, axis=0, keepdims=True)

    U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)

    total_var = float(jnp.sum(S ** 2))
    explained = S ** 2 / max(total_var, 1e-10)

    n = min(n_components, len(S))
    cumulative = jnp.cumsum(explained)

    # How many dims for 90% variance?
    eff_dim_90 = int(jnp.searchsorted(cumulative, 0.9)) + 1

    return {
        'explained_variance': [float(v) for v in explained[:n]],
        'cumulative_variance': [float(c) for c in cumulative[:n]],
        'dims_for_90pct': eff_dim_90,
        'total_dimensions': len(S),
        'top_singular_value': float(S[0]),
    }
