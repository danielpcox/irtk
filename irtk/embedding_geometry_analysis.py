"""Embedding geometry analysis: geometric structure of the embedding space."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def embedding_isotropy(model: HookedTransformer) -> dict:
    """How isotropic is the embedding space?

    Perfectly isotropic = directions uniformly distributed.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    # Center embeddings
    centered = W_E - jnp.mean(W_E, axis=0, keepdims=True)
    # Covariance eigenvalues
    cov = (centered.T @ centered) / W_E.shape[0]
    eigvals = jnp.linalg.eigvalsh(cov)
    eigvals = jnp.maximum(eigvals, 0)
    eigvals = eigvals[::-1]  # descending
    total = jnp.sum(eigvals) + 1e-10
    normed = eigvals / total

    # Isotropy = exp(entropy) / d_model
    entropy = -jnp.sum(normed * jnp.log(normed + 1e-10))
    isotropy = float(jnp.exp(entropy)) / W_E.shape[1]

    # Top eigenvalue fraction
    top_fraction = float(eigvals[0] / total)

    return {
        'd_model': W_E.shape[1],
        'd_vocab': W_E.shape[0],
        'isotropy': float(isotropy),
        'top_eigenvalue_fraction': top_fraction,
        'effective_dimension': float(jnp.exp(entropy)),
        'mean_norm': float(jnp.mean(jnp.linalg.norm(W_E, axis=1))),
    }


def embedding_nearest_neighbors(model: HookedTransformer, token_ids: list | None = None,
                                 top_k: int = 5) -> dict:
    """Find nearest neighbors in embedding space for selected tokens."""
    W_E = model.embed.W_E  # [d_vocab, d_model]
    if token_ids is None:
        token_ids = list(range(min(10, W_E.shape[0])))

    norms = jnp.linalg.norm(W_E, axis=1, keepdims=True) + 1e-10
    normed = W_E / norms

    per_token = []
    for tid in token_ids:
        tid = int(tid)
        sims = normed @ normed[tid]  # [d_vocab]
        # Exclude self
        sims = sims.at[tid].set(-2.0)
        top_indices = jnp.argsort(sims)[-top_k:][::-1]

        neighbors = []
        for idx in top_indices:
            idx = int(idx)
            neighbors.append({
                'token_id': idx,
                'cosine': float(sims[idx]),
            })

        per_token.append({
            'token_id': tid,
            'norm': float(jnp.linalg.norm(W_E[tid])),
            'neighbors': neighbors,
        })

    return {
        'per_token': per_token,
    }


def embedding_pca_structure(model: HookedTransformer, n_components: int = 5) -> dict:
    """PCA of the embedding matrix — top principal components and variance explained."""
    W_E = model.embed.W_E  # [d_vocab, d_model]
    centered = W_E - jnp.mean(W_E, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)

    total_var = float(jnp.sum(S ** 2))
    components = []
    cumulative = 0.0
    for i in range(min(n_components, len(S))):
        var_i = float(S[i] ** 2)
        cumulative += var_i
        components.append({
            'component': i,
            'singular_value': float(S[i]),
            'variance_fraction': var_i / (total_var + 1e-10),
            'cumulative_variance': cumulative / (total_var + 1e-10),
        })

    return {
        'd_model': W_E.shape[1],
        'd_vocab': W_E.shape[0],
        'total_variance': total_var,
        'components': components,
    }


def embedding_cluster_structure(model: HookedTransformer, n_samples: int = 50) -> dict:
    """Analyze clustering tendency in embeddings via pairwise cosines."""
    W_E = model.embed.W_E  # [d_vocab, d_model]
    n = min(n_samples, W_E.shape[0])

    norms = jnp.linalg.norm(W_E[:n], axis=1, keepdims=True) + 1e-10
    normed = W_E[:n] / norms
    cos_matrix = normed @ normed.T

    # Stats over off-diagonal
    mask = 1 - jnp.eye(n)
    off_diag = cos_matrix * mask
    n_pairs = n * (n - 1)
    mean_cos = float(jnp.sum(off_diag) / n_pairs)
    max_cos = float(jnp.max(off_diag))
    min_cos = float(jnp.min(cos_matrix + jnp.eye(n) * 10) )  # offset diagonal

    # Fraction of high-similarity pairs
    high_sim = float(jnp.sum((off_diag > 0.5) * mask) / n_pairs)

    return {
        'n_tokens_sampled': n,
        'mean_pairwise_cosine': mean_cos,
        'max_pairwise_cosine': max_cos,
        'min_pairwise_cosine': min_cos,
        'fraction_high_similarity': high_sim,
        'is_well_spread': mean_cos < 0.3,
    }


def embedding_norm_distribution(model: HookedTransformer, top_k: int = 10) -> dict:
    """Distribution of embedding vector norms — outliers may be special tokens."""
    W_E = model.embed.W_E  # [d_vocab, d_model]
    norms = jnp.linalg.norm(W_E, axis=1)

    mean_norm = float(jnp.mean(norms))
    std_norm = float(jnp.std(norms))

    top_indices = jnp.argsort(norms)[-top_k:][::-1]
    top_tokens = []
    for idx in top_indices:
        idx = int(idx)
        top_tokens.append({
            'token_id': idx,
            'norm': float(norms[idx]),
            'z_score': float((norms[idx] - mean_norm) / (std_norm + 1e-10)),
        })

    bottom_indices = jnp.argsort(norms)[:top_k]
    bottom_tokens = []
    for idx in bottom_indices:
        idx = int(idx)
        bottom_tokens.append({
            'token_id': idx,
            'norm': float(norms[idx]),
            'z_score': float((norms[idx] - mean_norm) / (std_norm + 1e-10)),
        })

    return {
        'd_vocab': W_E.shape[0],
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'max_norm': float(jnp.max(norms)),
        'min_norm': float(jnp.min(norms)),
        'top_norm_tokens': top_tokens,
        'bottom_norm_tokens': bottom_tokens,
    }
