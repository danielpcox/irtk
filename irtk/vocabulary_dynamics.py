"""Token-level vocabulary analysis.

Analyzes how the model's vocabulary representation space is structured,
including frequency biases, embedding geometry, vocabulary subspaces,
and how different token types are treated.

References:
    Gao et al. (2019) "Representation Degeneration in Token Embeddings"
    Ethayarajh (2019) "How Contextual are Contextualized Word Representations?"
"""

import jax
import jax.numpy as jnp
import numpy as np


def embedding_unembed_alignment(model, top_k=10):
    """Measure alignment between embedding and unembedding for each token.

    Tokens with high alignment between their embedding and unembedding
    vectors are strongly self-promoting in the residual stream.

    Args:
        model: HookedTransformer model.
        top_k: Number of top/bottom aligned tokens to return.

    Returns:
        dict with:
            cosine_similarities: array [d_vocab] of per-token alignment
            mean_alignment: float, average alignment
            top_aligned_tokens: array of top_k most aligned token indices
            bottom_aligned_tokens: array of top_k least aligned token indices
            alignment_std: float, standard deviation of alignment
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    d_vocab = W_E.shape[0]

    # Compute cosine similarity for each token
    embed_norms = jnp.linalg.norm(W_E, axis=1) + 1e-10  # [d_vocab]
    unembed_norms = jnp.linalg.norm(W_U, axis=0) + 1e-10  # [d_vocab]

    # dot product of each token's embedding with its unembedding column
    dots = jnp.sum(W_E * W_U.T, axis=1)  # [d_vocab]
    cosines = dots / (embed_norms * unembed_norms)

    cosines_np = np.array(cosines)
    sorted_idx = np.argsort(cosines_np)

    return {
        "cosine_similarities": cosines_np,
        "mean_alignment": float(np.mean(cosines_np)),
        "top_aligned_tokens": sorted_idx[-top_k:][::-1],
        "bottom_aligned_tokens": sorted_idx[:top_k],
        "alignment_std": float(np.std(cosines_np)),
    }


def vocab_subspace_analysis(model, n_components=5):
    """Analyze the principal subspace of the vocabulary embedding space.

    Uses SVD to find the dominant directions in the embedding matrix
    and measure how much variance they capture.

    Args:
        model: HookedTransformer model.
        n_components: Number of principal components to analyze.

    Returns:
        dict with:
            singular_values: array of top singular values
            explained_variance_ratio: array of variance ratios per component
            cumulative_variance: array of cumulative variance explained
            effective_rank: float, effective dimensionality
            mean_embedding_norm: float
    """
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]

    # Center embeddings
    mean_embed = np.mean(W_E, axis=0, keepdims=True)
    centered = W_E - mean_embed

    # SVD
    n_comp = min(n_components, min(centered.shape))
    # Use truncated SVD via numpy
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    S = S[:n_comp]

    total_var = np.sum(S ** 2)
    var_ratios = (S ** 2) / (total_var + 1e-10)
    cumulative = np.cumsum(var_ratios)

    # Effective rank (entropy-based)
    all_S = np.linalg.svd(centered, compute_uv=False)
    all_var = (all_S ** 2) / (np.sum(all_S ** 2) + 1e-10)
    all_var = all_var[all_var > 1e-12]
    effective_rank = float(np.exp(-np.sum(all_var * np.log(all_var + 1e-12))))

    norms = np.linalg.norm(W_E, axis=1)

    return {
        "singular_values": S.astype(np.float64),
        "explained_variance_ratio": var_ratios.astype(np.float64),
        "cumulative_variance": cumulative.astype(np.float64),
        "effective_rank": effective_rank,
        "mean_embedding_norm": float(np.mean(norms)),
    }


def token_frequency_bias(model, top_k=10):
    """Analyze bias toward high-norm tokens in the unembedding.

    Tokens with large unembedding norms have an inherent logit advantage
    regardless of context. This measures that bias.

    Args:
        model: HookedTransformer model.
        top_k: Number of top/bottom tokens to return.

    Returns:
        dict with:
            unembed_norms: array [d_vocab] of per-token unembedding norms
            mean_norm: float
            norm_std: float
            highest_norm_tokens: array of top_k tokens with highest norms
            lowest_norm_tokens: array of top_k tokens with lowest norms
            norm_ratio: float, ratio of max to min norm
    """
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    norms = np.linalg.norm(W_U, axis=0)  # [d_vocab]
    sorted_idx = np.argsort(norms)

    min_norm = float(norms[sorted_idx[0]])
    max_norm = float(norms[sorted_idx[-1]])

    return {
        "unembed_norms": norms,
        "mean_norm": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "highest_norm_tokens": sorted_idx[-top_k:][::-1],
        "lowest_norm_tokens": sorted_idx[:top_k],
        "norm_ratio": max_norm / (min_norm + 1e-10),
    }


def embedding_isotropy(model, n_samples=None):
    """Measure the isotropy of the embedding space.

    A perfectly isotropic embedding space has uniform angular distribution.
    Degenerate spaces have tokens clustered in a low-dimensional subspace.

    Args:
        model: HookedTransformer model.
        n_samples: Number of token pairs to sample (None = all pairs for small vocab).

    Returns:
        dict with:
            mean_cosine: float, average cosine similarity between random pairs
            std_cosine: float, standard deviation
            min_cosine: float
            max_cosine: float
            isotropy_score: float, 1 - |mean_cosine| (1 = perfectly isotropic)
    """
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    norms = np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-10
    W_E_normed = W_E / norms

    d_vocab = W_E.shape[0]

    if n_samples is None and d_vocab <= 200:
        # Compute all pairs
        sim_matrix = W_E_normed @ W_E_normed.T
        # Extract upper triangle (exclude diagonal)
        mask = np.triu(np.ones((d_vocab, d_vocab), dtype=bool), k=1)
        cosines = sim_matrix[mask]
    else:
        # Sample pairs
        n_s = n_samples if n_samples is not None else 5000
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, d_vocab, size=n_s)
        idx_b = rng.randint(0, d_vocab, size=n_s)
        # Ensure different tokens
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        cosines = np.sum(W_E_normed[idx_a] * W_E_normed[idx_b], axis=1)

    return {
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "isotropy_score": float(1.0 - abs(np.mean(cosines))),
    }


def token_neighborhood_structure(model, query_tokens, k=5):
    """Find nearest neighbors for specified tokens in embedding space.

    Useful for understanding what semantic relationships the embedding
    space captures.

    Args:
        model: HookedTransformer model.
        query_tokens: array of token indices to query.
        k: Number of nearest neighbors.

    Returns:
        dict with:
            neighbors: dict mapping token_idx -> array of k nearest neighbor indices
            neighbor_similarities: dict mapping token_idx -> array of k cosine similarities
            mean_neighbor_similarity: float, average nearest-neighbor similarity
            self_similarity_rank: dict mapping token_idx -> rank of self in unembed space
    """
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    norms = np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-10
    W_E_normed = W_E / norms

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    unembed_norms = np.linalg.norm(W_U, axis=0, keepdims=True) + 1e-10
    W_U_normed = W_U / unembed_norms

    neighbors = {}
    neighbor_sims = {}
    self_ranks = {}
    all_nn_sims = []

    for token_idx in query_tokens:
        token_idx = int(token_idx)
        query_vec = W_E_normed[token_idx]  # [d_model]

        # Cosine similarity to all other embeddings
        sims = W_E_normed @ query_vec  # [d_vocab]
        sims[token_idx] = -2.0  # Exclude self

        top_indices = np.argsort(sims)[::-1][:k]
        neighbors[token_idx] = top_indices
        neighbor_sims[token_idx] = sims[top_indices]
        all_nn_sims.extend(sims[top_indices].tolist())

        # Self-similarity: rank of this token in its own unembed predictions
        unembed_sims = W_U_normed.T @ query_vec  # [d_vocab]
        rank = int(np.sum(unembed_sims > unembed_sims[token_idx]))
        self_ranks[token_idx] = rank

    return {
        "neighbors": neighbors,
        "neighbor_similarities": neighbor_sims,
        "mean_neighbor_similarity": float(np.mean(all_nn_sims)) if all_nn_sims else 0.0,
        "self_similarity_rank": self_ranks,
    }
