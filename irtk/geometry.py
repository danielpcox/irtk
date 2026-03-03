"""Representation geometry analysis.

Tools for analyzing the geometric structure of model representations:
- representational_similarity: CKA between layers
- subspace_overlap: Overlap between activation subspaces
- intrinsic_dimensionality: Estimate effective dimensionality
- layer_similarity_matrix: Pairwise similarity between all layers
- representation_drift: How representations change across layers
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def _collect_activations(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    hook_name: str,
    pos: int = -1,
) -> np.ndarray:
    """Collect activations at a hook point across sequences.

    Returns [n_sequences, d_model].
    """
    activations = []
    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        act = np.array(cache[hook_name])[pos]  # [d_model]
        activations.append(act)
    if not activations:
        return np.zeros((0, model.cfg.d_model))
    return np.stack(activations)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (Centered Kernel Alignment) between two representations.

    Args:
        X: [n, d1] activations from first layer.
        Y: [n, d2] activations from second layer.

    Returns:
        CKA similarity in [0, 1].
    """
    n = X.shape[0]
    if n < 2:
        return 0.0

    # Center
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Linear kernels
    # HSIC(X,Y) = ||Y^T X||_F^2 / (n-1)^2
    XtY = X.T @ Y
    hsic_xy = np.sum(XtY ** 2)

    XtX = X.T @ X
    hsic_xx = np.sum(XtX ** 2)

    YtY = Y.T @ Y
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def representational_similarity(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    hook_name_a: str,
    hook_name_b: str,
    pos: int = -1,
    method: str = "cka",
) -> float:
    """Compute representational similarity between two layers.

    Args:
        model: HookedTransformer.
        token_sequences: List of input token sequences.
        hook_name_a: First hook point.
        hook_name_b: Second hook point.
        pos: Sequence position to analyze (-1 for last).
        method: "cka" for CKA, "cosine" for average cosine similarity.

    Returns:
        Similarity score.
    """
    X = _collect_activations(model, token_sequences, hook_name_a, pos)
    Y = _collect_activations(model, token_sequences, hook_name_b, pos)

    if method == "cka":
        return _linear_cka(X, Y)
    elif method == "cosine":
        # Average cosine similarity between paired representations
        norms_x = np.linalg.norm(X, axis=-1, keepdims=True)
        norms_y = np.linalg.norm(Y, axis=-1, keepdims=True)
        norms_x = np.maximum(norms_x, 1e-10)
        norms_y = np.maximum(norms_y, 1e-10)
        cosines = np.sum((X / norms_x) * (Y / norms_y), axis=-1)
        return float(np.mean(cosines))
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'cka' or 'cosine'.")


def subspace_overlap(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    hook_name_a: str,
    hook_name_b: str,
    n_dims: int = 10,
    pos: int = -1,
) -> float:
    """Measure overlap between activation subspaces at two layers.

    Computes the mean squared cosine similarity between the top principal
    components of the two activation sets.

    Args:
        model: HookedTransformer.
        token_sequences: List of input token sequences.
        hook_name_a: First hook point.
        hook_name_b: Second hook point.
        n_dims: Number of principal components to compare.
        pos: Sequence position to analyze.

    Returns:
        Overlap score in [0, 1]. 1 = identical subspaces.
    """
    X = _collect_activations(model, token_sequences, hook_name_a, pos)
    Y = _collect_activations(model, token_sequences, hook_name_b, pos)

    if X.shape[0] < 2 or Y.shape[0] < 2:
        return 0.0

    # PCA for each
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    _, _, Vx = np.linalg.svd(X_centered, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y_centered, full_matrices=False)

    k = min(n_dims, Vx.shape[0], Vy.shape[0])
    Vx = Vx[:k]
    Vy = Vy[:k]

    # Overlap = mean squared cosine between principal components
    # Using Grassmann distance proxy: ||Vx @ Vy^T||_F^2 / k
    cross = Vx @ Vy.T
    overlap = np.sum(cross ** 2) / k
    return float(overlap)


def intrinsic_dimensionality(
    activations: np.ndarray,
    method: str = "participation_ratio",
) -> float:
    """Estimate intrinsic dimensionality of activations.

    Args:
        activations: [n_samples, d_model] activation matrix.
        method: "participation_ratio" or "explained_variance_90".

    Returns:
        Estimated intrinsic dimensionality.
    """
    if activations.shape[0] < 2:
        return 0.0

    centered = activations - np.mean(activations, axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = S ** 2

    if method == "participation_ratio":
        sum_eig = np.sum(eigenvalues)
        sum_sq_eig = np.sum(eigenvalues ** 2)
        if sum_sq_eig < 1e-10:
            return 0.0
        return float(sum_eig ** 2 / sum_sq_eig)

    elif method == "explained_variance_90":
        total = np.sum(eigenvalues)
        if total < 1e-10:
            return 0.0
        cumsum = np.cumsum(eigenvalues) / total
        # Number of components to explain 90% variance
        return float(np.searchsorted(cumsum, 0.9) + 1)

    else:
        raise ValueError(f"Unknown method: {method!r}. "
                        "Choose 'participation_ratio' or 'explained_variance_90'.")


def layer_similarity_matrix(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    pos: int = -1,
    method: str = "cka",
) -> dict[str, np.ndarray]:
    """Pairwise representational similarity between all layers.

    Args:
        model: HookedTransformer.
        token_sequences: List of input token sequences.
        pos: Sequence position to analyze.
        method: "cka" or "cosine".

    Returns:
        Dict with:
        - "matrix": [n_layers+1, n_layers+1] similarity matrix
        - "labels": list of hook names
    """
    hooks = ["hook_embed"]
    for l in range(model.cfg.n_layers):
        hooks.append(f"blocks.{l}.hook_resid_post")
    n = len(hooks)

    # Collect all activations first
    all_acts = {}
    for hook in hooks:
        all_acts[hook] = _collect_activations(model, token_sequences, hook, pos)

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                X = all_acts[hooks[i]]
                Y = all_acts[hooks[j]]
                if method == "cka":
                    sim = _linear_cka(X, Y)
                elif method == "cosine":
                    norms_x = np.linalg.norm(X, axis=-1, keepdims=True)
                    norms_y = np.linalg.norm(Y, axis=-1, keepdims=True)
                    norms_x = np.maximum(norms_x, 1e-10)
                    norms_y = np.maximum(norms_y, 1e-10)
                    cosines = np.sum((X / norms_x) * (Y / norms_y), axis=-1)
                    sim = float(np.mean(cosines))
                else:
                    raise ValueError(f"Unknown method: {method!r}.")
                matrix[i, j] = sim
                matrix[j, i] = sim

    labels = ["embed"] + [f"block_{l}" for l in range(model.cfg.n_layers)]
    return {"matrix": matrix, "labels": labels}


def representation_drift(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    pos: int = -1,
) -> dict[str, np.ndarray]:
    """Measure how representations change across layers.

    For each consecutive pair of layers, computes the average L2 distance
    and cosine similarity between representations.

    Args:
        model: HookedTransformer.
        token_sequences: List of input token sequences.
        pos: Sequence position to analyze.

    Returns:
        Dict with:
        - "l2_distances": [n_layers] L2 distance between consecutive layers
        - "cosine_similarities": [n_layers] cosine similarity between consecutive layers
        - "labels": list of transition labels
    """
    hooks = ["hook_embed"]
    for l in range(model.cfg.n_layers):
        hooks.append(f"blocks.{l}.hook_resid_post")

    all_acts = {}
    for hook in hooks:
        all_acts[hook] = _collect_activations(model, token_sequences, hook, pos)

    n_transitions = len(hooks) - 1
    l2_dists = np.zeros(n_transitions)
    cosine_sims = np.zeros(n_transitions)

    for i in range(n_transitions):
        X = all_acts[hooks[i]]
        Y = all_acts[hooks[i + 1]]

        # L2 distance
        diffs = Y - X
        l2_dists[i] = float(np.mean(np.linalg.norm(diffs, axis=-1)))

        # Cosine similarity
        norms_x = np.linalg.norm(X, axis=-1)
        norms_y = np.linalg.norm(Y, axis=-1)
        denom = np.maximum(norms_x * norms_y, 1e-10)
        cosines = np.sum(X * Y, axis=-1) / denom
        cosine_sims[i] = float(np.mean(cosines))

    labels = [f"{'embed' if i == 0 else f'block_{i-1}'}->block_{i}"
              for i in range(n_transitions)]

    return {
        "l2_distances": l2_dists,
        "cosine_similarities": cosine_sims,
        "labels": labels,
    }
