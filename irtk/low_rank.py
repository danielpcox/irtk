"""Low-rank analysis and approximation of model weight matrices.

Tools for understanding the effective dimensionality of model components:
- weight_svd: SVD of any named weight matrix
- effective_rank: How many singular values carry most of the energy
- low_rank_approximation: Truncated SVD reconstruction of weight matrices
- weight_spectrum_similarity: Compare spectral structure across layers/heads
- apply_low_rank_weights: Replace a weight matrix with its rank-k approximation
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer


def weight_svd(
    model: HookedTransformer,
    weight_path: str,
) -> dict:
    """Compute SVD of a model weight matrix.

    Args:
        model: HookedTransformer.
        weight_path: Dot-separated path to the weight, e.g.
            "blocks.0.attn.W_Q" (returns [n_heads, d_model, d_head]),
            "blocks.0.mlp.W_in" (returns [d_model, d_mlp]),
            "unembed.W_U" (returns [d_model, d_vocab]).

    Returns:
        Dict with:
        - "U": left singular vectors
        - "S": singular values (descending)
        - "Vh": right singular vectors (conjugate transposed)
        - "shape": original weight shape
        - "rank": numerical rank (S > 1e-6 * S[0])
    """
    W = _get_weight(model, weight_path)
    W_np = np.array(W)

    # Flatten to 2D if higher-dimensional (e.g. [n_heads, d_model, d_head])
    original_shape = W_np.shape
    if W_np.ndim > 2:
        W_2d = W_np.reshape(-1, W_np.shape[-1])
    else:
        W_2d = W_np

    U, S, Vh = np.linalg.svd(W_2d, full_matrices=False)
    rank = int(np.sum(S > 1e-6 * S[0]))

    return {
        "U": U,
        "S": S,
        "Vh": Vh,
        "shape": original_shape,
        "rank": rank,
    }


def effective_rank(
    model: HookedTransformer,
    weight_path: str,
    threshold: float = 0.99,
) -> dict:
    """Compute the effective rank of a weight matrix.

    The effective rank is the minimum number of singular values
    needed to capture `threshold` fraction of the total energy
    (sum of squared singular values = Frobenius norm squared).

    Args:
        model: HookedTransformer.
        weight_path: Dot-separated path to the weight.
        threshold: Fraction of energy to capture (default 0.99).

    Returns:
        Dict with:
        - "effective_rank": number of SVs for the threshold
        - "full_rank": numerical rank
        - "total_energy": sum of S^2
        - "cumulative_energy": cumulative fraction of energy per SV
        - "singular_values": all singular values
    """
    svd = weight_svd(model, weight_path)
    S = svd["S"]
    S2 = S ** 2
    total = S2.sum()
    cumulative = np.cumsum(S2) / max(total, 1e-30)

    eff_rank = int(np.searchsorted(cumulative, threshold)) + 1
    eff_rank = min(eff_rank, len(S))

    return {
        "effective_rank": eff_rank,
        "full_rank": svd["rank"],
        "total_energy": float(total),
        "cumulative_energy": cumulative,
        "singular_values": S,
    }


def low_rank_approximation(
    model: HookedTransformer,
    weight_path: str,
    rank: int,
) -> dict:
    """Compute rank-k approximation of a weight matrix.

    Uses truncated SVD: W_k = U[:, :k] @ diag(S[:k]) @ Vh[:k, :].

    Args:
        model: HookedTransformer.
        weight_path: Dot-separated path to the weight.
        rank: Number of singular values to keep.

    Returns:
        Dict with:
        - "approximation": the rank-k weight matrix (original shape)
        - "reconstruction_error": Frobenius norm of (W - W_k)
        - "relative_error": error / ||W||_F
        - "energy_captured": fraction of squared singular values retained
    """
    svd = weight_svd(model, weight_path)
    U, S, Vh = svd["U"], svd["S"], svd["Vh"]
    original_shape = svd["shape"]

    k = min(rank, len(S))
    W_k = (U[:, :k] * S[:k]) @ Vh[:k, :]

    # Reshape back if needed
    if len(original_shape) > 2:
        W_k = W_k.reshape(original_shape)

    W = np.array(_get_weight(model, weight_path))
    error = float(np.linalg.norm(W.reshape(-1, W.shape[-1]) - (U[:, :k] * S[:k]) @ Vh[:k, :]))
    W_norm = float(np.linalg.norm(W))
    energy_captured = float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-30))

    return {
        "approximation": W_k,
        "reconstruction_error": error,
        "relative_error": error / max(W_norm, 1e-30),
        "energy_captured": energy_captured,
    }


def weight_spectrum_similarity(
    model: HookedTransformer,
    path_a: str,
    path_b: str,
) -> dict:
    """Compare spectral structure of two weight matrices.

    Computes cosine similarity between normalized singular value spectra.
    Useful for comparing how different layers or heads use their capacity.

    Args:
        model: HookedTransformer.
        path_a: First weight path.
        path_b: Second weight path.

    Returns:
        Dict with:
        - "spectral_similarity": cosine similarity of normalized SV spectra
        - "rank_a": effective rank of A
        - "rank_b": effective rank of B
        - "spectrum_a": normalized singular values of A
        - "spectrum_b": normalized singular values of B
    """
    svd_a = weight_svd(model, path_a)
    svd_b = weight_svd(model, path_b)

    S_a = svd_a["S"]
    S_b = svd_b["S"]

    # Normalize and pad to same length
    n = max(len(S_a), len(S_b))
    spec_a = np.zeros(n)
    spec_b = np.zeros(n)
    spec_a[:len(S_a)] = S_a / max(np.linalg.norm(S_a), 1e-30)
    spec_b[:len(S_b)] = S_b / max(np.linalg.norm(S_b), 1e-30)

    similarity = float(np.dot(spec_a, spec_b))

    return {
        "spectral_similarity": similarity,
        "rank_a": svd_a["rank"],
        "rank_b": svd_b["rank"],
        "spectrum_a": spec_a,
        "spectrum_b": spec_b,
    }


def apply_low_rank_weights(
    model: HookedTransformer,
    weight_path: str,
    rank: int,
) -> HookedTransformer:
    """Replace a weight matrix with its rank-k approximation.

    Returns a new model (original is unchanged).

    Args:
        model: HookedTransformer.
        weight_path: Dot-separated path to the weight.
        rank: Number of singular values to keep.

    Returns:
        New HookedTransformer with the approximated weight.
    """
    approx = low_rank_approximation(model, weight_path, rank)
    W_k = jnp.array(approx["approximation"])

    # Navigate the path to set via eqx.tree_at
    parts = weight_path.split(".")
    def get_fn(m):
        obj = m
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    return eqx.tree_at(get_fn, model, W_k)


def _get_weight(model: HookedTransformer, weight_path: str) -> jnp.ndarray:
    """Navigate a dot-separated path to get a weight tensor."""
    obj = model
    for part in weight_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
