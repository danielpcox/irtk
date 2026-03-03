"""Token embedding analysis utilities.

Tools for analyzing token embedding and unembedding spaces:
- embedding_similarity: Cosine similarity between token embeddings
- nearest_neighbors: Find k nearest tokens to a token or direction
- embedding_pca: PCA of token embedding space
- token_analogy: Solve analogies in embedding space
- embedding_cluster: Cluster tokens by embedding similarity
- embed_unembed_alignment: Measure W_E / W_U alignment per token
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def embedding_similarity(
    model: HookedTransformer,
    token_a: int,
    token_b: int,
    space: str = "embed",
) -> float:
    """Cosine similarity between two token embeddings.

    Args:
        model: HookedTransformer.
        token_a: First token ID.
        token_b: Second token ID.
        space: "embed" for W_E, "unembed" for W_U.

    Returns:
        Cosine similarity in [-1, 1].
    """
    if space == "embed":
        W = np.array(model.embed.W_E)
    elif space == "unembed":
        W = np.array(model.unembed.W_U).T  # [d_vocab, d_model]
    else:
        raise ValueError(f"Unknown space: {space!r}. Choose 'embed' or 'unembed'.")

    va = W[token_a]
    vb = W[token_b]
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def nearest_neighbors(
    model: HookedTransformer,
    query: int | np.ndarray,
    k: int = 10,
    space: str = "embed",
) -> list[tuple[int, float]]:
    """Find k nearest tokens to a query token or direction.

    Args:
        model: HookedTransformer.
        query: Token ID (int) or direction vector [d_model].
        k: Number of neighbors to return.
        space: "embed" for W_E, "unembed" for W_U.

    Returns:
        List of (token_id, cosine_similarity) sorted by similarity descending.
    """
    if space == "embed":
        W = np.array(model.embed.W_E)
    elif space == "unembed":
        W = np.array(model.unembed.W_U).T
    else:
        raise ValueError(f"Unknown space: {space!r}. Choose 'embed' or 'unembed'.")

    if isinstance(query, (int, np.integer)):
        vec = W[query]
    else:
        vec = np.array(query)

    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-10:
        return [(0, 0.0)] * k

    norms = np.linalg.norm(W, axis=-1)
    norms = np.maximum(norms, 1e-10)
    similarities = (W @ vec) / (norms * vec_norm)

    top_indices = np.argsort(similarities)[::-1][:k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def embedding_pca(
    model: HookedTransformer,
    token_ids: Optional[list[int]] = None,
    n_components: int = 2,
    space: str = "embed",
) -> dict[str, np.ndarray]:
    """PCA of token embedding space.

    Args:
        model: HookedTransformer.
        token_ids: Subset of tokens to include. If None, uses all.
        n_components: Number of PCA components.
        space: "embed" for W_E, "unembed" for W_U.

    Returns:
        Dict with:
        - "projections": [n_tokens, n_components] PCA projections
        - "components": [n_components, d_model] principal components
        - "explained_variance": [n_components] variance ratios
        - "token_ids": [n_tokens] token IDs used
    """
    if space == "embed":
        W = np.array(model.embed.W_E)
    elif space == "unembed":
        W = np.array(model.unembed.W_U).T
    else:
        raise ValueError(f"Unknown space: {space!r}. Choose 'embed' or 'unembed'.")

    if token_ids is not None:
        W = W[token_ids]
        ids = np.array(token_ids)
    else:
        ids = np.arange(W.shape[0])

    # Center
    mean = np.mean(W, axis=0)
    centered = W - mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = np.sum(S ** 2)
    explained = S[:n_components] ** 2 / (total_var + 1e-10)

    projections = centered @ Vt[:n_components].T
    components = Vt[:n_components]

    return {
        "projections": projections,
        "components": components,
        "explained_variance": explained,
        "token_ids": ids,
    }


def token_analogy(
    model: HookedTransformer,
    a: int,
    b: int,
    c: int,
    k: int = 5,
    space: str = "embed",
) -> list[tuple[int, float]]:
    """Solve embedding analogy: a is to b as c is to ?

    Computes b - a + c and finds nearest neighbors.

    Args:
        model: HookedTransformer.
        a, b, c: Token IDs for the analogy a:b :: c:?
        k: Number of results.
        space: "embed" for W_E, "unembed" for W_U.

    Returns:
        List of (token_id, similarity) for the answer.
    """
    if space == "embed":
        W = np.array(model.embed.W_E)
    elif space == "unembed":
        W = np.array(model.unembed.W_U).T
    else:
        raise ValueError(f"Unknown space: {space!r}. Choose 'embed' or 'unembed'.")

    direction = W[b] - W[a] + W[c]
    # Exclude the input tokens from results
    exclude = {a, b, c}
    results = nearest_neighbors(model, direction, k=k + len(exclude), space=space)
    return [(tid, sim) for tid, sim in results if tid not in exclude][:k]


def embedding_cluster(
    model: HookedTransformer,
    token_ids: list[int],
    n_clusters: int = 5,
    space: str = "embed",
) -> dict[str, np.ndarray]:
    """Cluster tokens by embedding similarity using k-means.

    Args:
        model: HookedTransformer.
        token_ids: Token IDs to cluster.
        n_clusters: Number of clusters.
        space: "embed" for W_E, "unembed" for W_U.

    Returns:
        Dict with:
        - "labels": [n_tokens] cluster assignment
        - "centroids": [n_clusters, d_model] cluster centers
        - "token_ids": [n_tokens] token IDs
    """
    if space == "embed":
        W = np.array(model.embed.W_E)
    elif space == "unembed":
        W = np.array(model.unembed.W_U).T
    else:
        raise ValueError(f"Unknown space: {space!r}. Choose 'embed' or 'unembed'.")

    embeddings = W[token_ids]
    n = len(token_ids)

    # Normalize
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0) + 1e-10
    normed = (embeddings - mean) / std

    # K-means
    rng = np.random.RandomState(42)
    actual_k = min(n_clusters, n)
    centers = normed[rng.choice(n, actual_k, replace=False)]

    for _ in range(50):
        dists = np.sum((normed[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        for k in range(actual_k):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = np.mean(normed[mask], axis=0)
            else:
                new_centers[k] = centers[k]

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return {
        "labels": labels,
        "centroids": centers,
        "token_ids": np.array(token_ids),
    }


def embed_unembed_alignment(
    model: HookedTransformer,
    token_ids: Optional[list[int]] = None,
) -> np.ndarray:
    """Measure alignment between embedding and unembedding vectors per token.

    For each token, computes cosine similarity between its W_E row and W_U column.
    High alignment means the model uses similar representations for reading and writing.

    Args:
        model: HookedTransformer.
        token_ids: Subset of tokens. If None, computes for all tokens.

    Returns:
        [n_tokens] array of cosine similarities.
    """
    W_E = np.array(model.embed.W_E)      # [d_vocab, d_model]
    W_U = np.array(model.unembed.W_U).T  # [d_vocab, d_model]

    if token_ids is not None:
        W_E = W_E[token_ids]
        W_U = W_U[token_ids]

    # Row-wise cosine similarity
    norm_E = np.linalg.norm(W_E, axis=-1)
    norm_U = np.linalg.norm(W_U, axis=-1)
    denom = np.maximum(norm_E * norm_U, 1e-10)
    similarities = np.sum(W_E * W_U, axis=-1) / denom

    return similarities
