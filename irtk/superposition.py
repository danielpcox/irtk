"""Superposition analysis utilities.

Tools for studying how models represent more features than dimensions:
- compute_feature_directions: Extract feature directions (PCA, random)
- feature_interference: Measure feature cosine similarity / interference
- dimensionality_analysis: Effective dimensionality via participation ratio
- activation_covariance: Covariance structure of activations
- feature_sparsity: Measure sparsity of activation patterns
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


def compute_feature_directions(
    activations: np.ndarray,
    n_features: Optional[int] = None,
    method: str = "pca",
) -> np.ndarray:
    """Extract feature directions from activation data.

    Args:
        activations: [n_samples, d_model] activation vectors.
        n_features: Number of feature directions to extract.
            Defaults to d_model.
        method: "pca" for principal components, "random" for random projections.

    Returns:
        [n_features, d_model] matrix of unit feature directions.
    """
    activations = np.asarray(activations)
    d_model = activations.shape[1]
    if n_features is None:
        n_features = d_model

    if method == "pca":
        # Center the data
        mean = np.mean(activations, axis=0)
        centered = activations - mean

        # Compute SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        directions = Vt[:n_features]  # [n_features, d_model]

        # Normalize
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-10)
        return directions

    elif method == "random":
        rng = np.random.RandomState(42)
        directions = rng.randn(n_features, d_model)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions / np.maximum(norms, 1e-10)

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: 'pca', 'random'")


def feature_interference(
    directions: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine similarity between feature directions.

    In a model with superposition, features that interfere with each
    other (high cosine similarity) compete for the same dimensions.

    Args:
        directions: [n_features, d_model] unit direction vectors.

    Returns:
        [n_features, n_features] cosine similarity matrix.
        Diagonal is 1.0, off-diagonal measures interference.
    """
    directions = np.asarray(directions)
    # Normalize to be safe
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    normed = directions / np.maximum(norms, 1e-10)
    return normed @ normed.T


def dimensionality_analysis(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    pos: int = -1,
) -> dict[str, np.ndarray]:
    """Compute effective dimensionality of activations at each layer.

    Uses the participation ratio: (sum of eigenvalues)^2 / (sum of eigenvalues^2).
    This equals d_model when all eigenvalues are equal (full rank) and 1 when
    one eigenvalue dominates.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        pos: Position to analyze (-1 for last).

    Returns:
        Dict with:
        - "participation_ratio": [n_layers+1] effective dimensionality per layer
        - "eigenvalue_spectra": list of [d_model] eigenvalue arrays per layer
        - "labels": layer names
    """
    n_layers = model.cfg.n_layers

    # Collect activations at each layer
    layer_activations = {}
    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        resid_stack = cache.accumulated_resid()  # [n_layers+1, seq, d_model]
        for i in range(resid_stack.shape[0]):
            if i not in layer_activations:
                layer_activations[i] = []
            layer_activations[i].append(np.array(resid_stack[i, pos]))

    n_components = len(layer_activations)
    participation_ratios = np.zeros(n_components)
    eigenvalue_spectra = []
    labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    for i in range(n_components):
        acts = np.stack(layer_activations[i])  # [n_samples, d_model]
        centered = acts - np.mean(acts, axis=0)
        cov = centered.T @ centered / max(len(acts) - 1, 1)  # [d_model, d_model]

        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        eigenvalue_spectra.append(eigenvalues)

        # Participation ratio
        sum_eig = np.sum(eigenvalues)
        sum_eig_sq = np.sum(eigenvalues ** 2)
        if sum_eig_sq > 1e-20:
            participation_ratios[i] = (sum_eig ** 2) / sum_eig_sq
        else:
            participation_ratios[i] = 0.0

    return {
        "participation_ratio": participation_ratios,
        "eigenvalue_spectra": eigenvalue_spectra,
        "labels": labels[:n_components],
    }


def activation_covariance(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    hook_name: str,
    pos: int = -1,
) -> np.ndarray:
    """Compute the covariance matrix of activations at a hook point.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        hook_name: Hook point to collect activations from.
        pos: Position to analyze (-1 for last).

    Returns:
        [d, d] covariance matrix where d is the activation dimension.
    """
    all_acts = []
    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name][pos])
            all_acts.append(act)

    if not all_acts:
        return np.zeros((model.cfg.d_model, model.cfg.d_model))

    acts = np.stack(all_acts)  # [n_samples, d]
    centered = acts - np.mean(acts, axis=0)
    return centered.T @ centered / max(len(acts) - 1, 1)


def feature_sparsity(
    activations: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Measure sparsity characteristics of activation patterns.

    Args:
        activations: [n_samples, d] activation vectors.
        threshold: Threshold below which activations are considered "off".

    Returns:
        Dict with:
        - "l0_mean": Average number of active dimensions per sample
        - "l0_fraction": Fraction of dimensions active on average
        - "kurtosis_mean": Mean excess kurtosis (higher = more peaked/sparse)
        - "gini_mean": Mean Gini coefficient (higher = more sparse)
    """
    activations = np.asarray(activations)
    n_samples, d = activations.shape

    # L0 sparsity
    active = np.abs(activations) > threshold
    l0_per_sample = np.sum(active, axis=-1)  # [n_samples]
    l0_mean = float(np.mean(l0_per_sample))
    l0_fraction = l0_mean / d

    # Kurtosis per dimension
    mean = np.mean(activations, axis=0)
    std = np.std(activations, axis=0) + 1e-10
    z = (activations - mean) / std
    kurtosis = np.mean(z ** 4, axis=0) - 3.0  # excess kurtosis
    kurtosis_mean = float(np.mean(kurtosis))

    # Gini coefficient per sample
    gini_values = []
    for i in range(min(n_samples, 1000)):  # cap for efficiency
        sorted_abs = np.sort(np.abs(activations[i]))
        n = len(sorted_abs)
        if n == 0 or np.sum(sorted_abs) < 1e-10:
            gini_values.append(0.0)
            continue
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_abs) / (n * np.sum(sorted_abs))) - (n + 1) / n
        gini_values.append(float(gini))

    return {
        "l0_mean": l0_mean,
        "l0_fraction": l0_fraction,
        "kurtosis_mean": kurtosis_mean,
        "gini_mean": float(np.mean(gini_values)) if gini_values else 0.0,
    }
