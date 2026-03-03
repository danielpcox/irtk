"""Activation distribution statistics.

Statistical analysis of activation distributions at each layer — moments,
heavy tails, normality, sparsity patterns, and multimodality. Understanding
these distributions is key to quantization, pruning, and identifying
anomalous computation.

Functions:
- layer_activation_moments: Mean, variance, skewness, kurtosis per layer
- kurtosis_profile: Track heavy-tailedness through the network
- normality_test_by_layer: Test how Gaussian activations are at each layer
- activation_sparsity_pattern: Measure sparsity structure per layer
- multimodality_detection: Detect multi-modal activation distributions

References:
    - Dettmers et al. (2022) "LLM.int8(): 8-bit Matrix Multiplication"
    - Sun et al. (2024) "Massive Activations in Large Language Models"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_activation_moments(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Compute mean, variance, skewness, kurtosis per layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "means": [n_layers] mean activation per layer
            "variances": [n_layers] variance per layer
            "skewnesses": [n_layers] skewness per layer
            "kurtoses": [n_layers] excess kurtosis per layer (0 = Gaussian)
            "layer_names": list of layer hook names
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    means = np.zeros(n_layers)
    variances = np.zeros(n_layers)
    skewnesses = np.zeros(n_layers)
    kurtoses = np.zeros(n_layers)
    names = []

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        names.append(key)
        if key in cache.cache_dict:
            act = np.array(cache.cache_dict[key]).flatten()
            m = float(np.mean(act))
            v = float(np.var(act))
            means[l] = m
            variances[l] = v

            if v > 1e-10:
                centered = act - m
                std = np.sqrt(v)
                skewnesses[l] = float(np.mean((centered / std) ** 3))
                kurtoses[l] = float(np.mean((centered / std) ** 4) - 3.0)  # excess

    return {
        "means": means,
        "variances": variances,
        "skewnesses": skewnesses,
        "kurtoses": kurtoses,
        "layer_names": names,
    }


def kurtosis_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    per_dimension: bool = False,
) -> dict:
    """Track heavy-tailedness through the network.

    High kurtosis indicates outlier dimensions; low kurtosis indicates
    uniform or platykurtic distributions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        per_dimension: If True, return per-dimension kurtosis.

    Returns:
        Dict with:
            "layer_kurtosis": [n_layers] overall kurtosis per layer
            "max_kurtosis_layer": layer with highest kurtosis
            "kurtosis_trend": whether kurtosis increases with depth ("increasing"/"decreasing"/"flat")
            "per_dim_kurtosis": [n_layers, d_model] if per_dimension=True, else None
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    layer_kurt = np.zeros(n_layers)
    per_dim = np.zeros((n_layers, d_model)) if per_dimension else None

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            act = np.array(cache.cache_dict[key])  # [seq, d_model]
            flat = act.flatten()
            m = np.mean(flat)
            std = np.std(flat)
            if std > 1e-10:
                layer_kurt[l] = float(np.mean(((flat - m) / std) ** 4) - 3.0)

            if per_dimension:
                for d in range(d_model):
                    col = act[:, d]
                    cm = np.mean(col)
                    cs = np.std(col)
                    if cs > 1e-10:
                        per_dim[l, d] = float(np.mean(((col - cm) / cs) ** 4) - 3.0)

    max_layer = int(np.argmax(np.abs(layer_kurt)))

    # Trend
    if n_layers >= 2:
        diff = layer_kurt[-1] - layer_kurt[0]
        if diff > 0.5:
            trend = "increasing"
        elif diff < -0.5:
            trend = "decreasing"
        else:
            trend = "flat"
    else:
        trend = "flat"

    return {
        "layer_kurtosis": layer_kurt,
        "max_kurtosis_layer": max_layer,
        "kurtosis_trend": trend,
        "per_dim_kurtosis": per_dim,
    }


def normality_test_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Test how Gaussian activations are at each layer.

    Uses the Jarque-Bera statistic (based on skewness and kurtosis) as
    a measure of non-Gaussianity.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "jb_statistics": [n_layers] Jarque-Bera statistic (0 = perfectly Gaussian)
            "most_gaussian_layer": layer closest to Gaussian
            "least_gaussian_layer": layer furthest from Gaussian
            "normality_scores": [n_layers] normalized normality (1 = Gaussian, 0 = very non-Gaussian)
    """
    moments = layer_activation_moments(model, tokens)
    n_layers = len(moments["skewnesses"])

    # Jarque-Bera: JB = (n/6) * (S^2 + K^2/4) where S=skewness, K=excess kurtosis
    jb = np.zeros(n_layers)
    seq_len = len(tokens)
    n_samples = seq_len * model.cfg.d_model

    for l in range(n_layers):
        s = moments["skewnesses"][l]
        k = moments["kurtoses"][l]
        jb[l] = (n_samples / 6.0) * (s ** 2 + (k ** 2) / 4.0)

    most_gaussian = int(np.argmin(jb))
    least_gaussian = int(np.argmax(jb))

    # Normality score: 1 / (1 + JB/n_samples) to normalize
    normality = 1.0 / (1.0 + jb / n_samples)

    return {
        "jb_statistics": jb,
        "most_gaussian_layer": most_gaussian,
        "least_gaussian_layer": least_gaussian,
        "normality_scores": normality,
    }


def activation_sparsity_pattern(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    threshold: float = 0.01,
) -> dict:
    """Measure sparsity structure per layer.

    Counts near-zero activations and measures the distribution of
    activation magnitudes.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        threshold: Magnitude below which an activation is "inactive".

    Returns:
        Dict with:
            "sparsity_ratios": [n_layers] fraction of near-zero activations
            "mean_magnitudes": [n_layers] mean absolute activation
            "l1_norms": [n_layers] L1 norm of activation (total magnitude)
            "sparsest_layer": layer with highest sparsity
            "densest_layer": layer with lowest sparsity
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    sparsity = np.zeros(n_layers)
    magnitudes = np.zeros(n_layers)
    l1_norms = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            act = np.array(cache.cache_dict[key])
            abs_act = np.abs(act)
            sparsity[l] = float(np.mean(abs_act < threshold))
            magnitudes[l] = float(np.mean(abs_act))
            l1_norms[l] = float(np.sum(abs_act))

    sparsest = int(np.argmax(sparsity))
    densest = int(np.argmin(sparsity))

    return {
        "sparsity_ratios": sparsity,
        "mean_magnitudes": magnitudes,
        "l1_norms": l1_norms,
        "sparsest_layer": sparsest,
        "densest_layer": densest,
    }


def multimodality_detection(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    n_bins: int = 20,
) -> dict:
    """Detect multi-modal activation distributions.

    Uses histogram-based analysis to detect whether activations at
    each layer follow a single mode or have multiple clusters.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        n_bins: Number of histogram bins.

    Returns:
        Dict with:
            "n_modes_per_layer": [n_layers] estimated number of modes
            "most_multimodal_layer": layer with most modes
            "bimodality_coefficients": [n_layers] bimodality coefficient (>0.555 suggests bimodal)
            "layer_histograms": list of (bin_centers, counts) per layer
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    n_modes = np.zeros(n_layers, dtype=int)
    bimodality = np.zeros(n_layers)
    histograms = []

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            act = np.array(cache.cache_dict[key]).flatten()

            # Histogram
            counts, edges = np.histogram(act, bins=n_bins)
            centers = (edges[:-1] + edges[1:]) / 2
            histograms.append((centers, counts))

            # Count modes: local maxima in histogram
            modes = 0
            for i in range(1, len(counts) - 1):
                if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                    modes += 1
            n_modes[l] = max(1, modes)

            # Bimodality coefficient: BC = (skew^2 + 1) / (kurt + 3)
            m = np.mean(act)
            std = np.std(act)
            if std > 1e-10:
                skew = float(np.mean(((act - m) / std) ** 3))
                kurt = float(np.mean(((act - m) / std) ** 4))
                bimodality[l] = (skew ** 2 + 1) / (kurt + 1e-10)
            else:
                bimodality[l] = 0.0
        else:
            histograms.append((np.zeros(n_bins), np.zeros(n_bins)))
            n_modes[l] = 1

    most_multimodal = int(np.argmax(n_modes))

    return {
        "n_modes_per_layer": n_modes,
        "most_multimodal_layer": most_multimodal,
        "bimodality_coefficients": bimodality,
        "layer_histograms": histograms,
    }
