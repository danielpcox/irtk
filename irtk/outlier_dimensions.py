"""Outlier dimension analysis: massive activations and dimension utilization.

Detects and analyzes outlier dimensions in the residual stream — dimensions
that carry disproportionately large activations. Also examines attention sink
tokens and how evenly the model utilizes its representational capacity.

Functions:
- detect_outlier_dimensions: Find dimensions with disproportionately large activations
- outlier_magnitude_across_layers: Track outlier dimensions through the network
- outlier_removal_effect: Measure impact of clamping outlier dimensions
- attention_sink_analysis: Analyze attention sink tokens and their activation patterns
- dimension_utilization_spectrum: How evenly the model uses its d_model dimensions

References:
    - Sun et al. (2024) "Massive Activations in Large Language Models"
    - Kovaleva et al. (2021) "BERT Busters: Outlier Dimensions"
    - Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def detect_outlier_dimensions(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = -1,
    threshold: float = 3.0,
) -> dict:
    """Find dimensions with disproportionately large activations.

    Identifies residual stream dimensions where the activation magnitude
    is significantly larger than the median, a pattern linked to model
    stability and numerical issues.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer to analyze (-1 = last).
        threshold: Number of MADs (median absolute deviations) for outlier.

    Returns:
        Dict with:
            "outlier_dims": list of (dim_index, magnitude) for outlier dimensions
            "magnitudes": [d_model] mean absolute activation per dimension
            "median_magnitude": median dimension magnitude
            "outlier_ratio": fraction of dimensions that are outliers
            "max_to_median_ratio": ratio of largest to median magnitude
    """
    _, cache = model.run_with_cache(tokens)

    if layer == -1:
        layer = model.cfg.n_layers - 1

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache.cache_dict:
        d = model.cfg.d_model
        return {
            "outlier_dims": [],
            "magnitudes": np.zeros(d),
            "median_magnitude": 0.0,
            "outlier_ratio": 0.0,
            "max_to_median_ratio": 1.0,
        }

    act = np.array(cache.cache_dict[key])  # [seq, d_model]
    magnitudes = np.mean(np.abs(act), axis=0)  # [d_model]

    median = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - median)))

    # Outlier detection using MAD
    outlier_mask = magnitudes > median + threshold * (mad + 1e-10)
    outlier_indices = np.where(outlier_mask)[0]
    outlier_dims = [(int(i), float(magnitudes[i])) for i in outlier_indices]
    outlier_dims.sort(key=lambda x: -x[1])

    ratio = float(len(outlier_indices) / len(magnitudes))
    max_to_med = float(np.max(magnitudes) / (median + 1e-10))

    return {
        "outlier_dims": outlier_dims,
        "magnitudes": magnitudes,
        "median_magnitude": median,
        "outlier_ratio": ratio,
        "max_to_median_ratio": max_to_med,
    }


def outlier_magnitude_across_layers(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_dims: Optional[list] = None,
) -> dict:
    """Track outlier dimension magnitudes through the network.

    For specified dimensions (or auto-detected outliers), tracks their
    magnitude at each layer to understand where outliers emerge.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_dims: Dimensions to track. Auto-detects if None.

    Returns:
        Dict with:
            "tracked_dims": list of tracked dimension indices
            "magnitude_trajectories": dict mapping dim -> [n_layers] magnitude curve
            "emergence_layers": dict mapping dim -> layer where magnitude first exceeds 2x median
            "growth_rates": dict mapping dim -> per-layer growth factor
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Auto-detect outliers from last layer if not specified
    if target_dims is None:
        result = detect_outlier_dimensions(model, tokens, layer=-1)
        target_dims = [d for d, _ in result["outlier_dims"][:5]]
        if not target_dims:
            # If no outliers, just track top-3 magnitude dims
            target_dims = list(np.argsort(result["magnitudes"])[::-1][:3])

    trajectories = {d: np.zeros(n_layers) for d in target_dims}
    layer_medians = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            act = np.array(cache.cache_dict[key])
            mags = np.mean(np.abs(act), axis=0)
            layer_medians[l] = float(np.median(mags))
            for d in target_dims:
                if d < len(mags):
                    trajectories[d][l] = float(mags[d])

    # Emergence layer: where magnitude first exceeds 2x median
    emergence = {}
    growth_rates = {}
    for d in target_dims:
        emergence[d] = -1
        for l in range(n_layers):
            if trajectories[d][l] > 2.0 * (layer_medians[l] + 1e-10):
                emergence[d] = l
                break

        # Growth rate
        if n_layers >= 2:
            rates = []
            for l in range(1, n_layers):
                if trajectories[d][l - 1] > 1e-10:
                    rates.append(trajectories[d][l] / trajectories[d][l - 1])
            growth_rates[d] = float(np.mean(rates)) if rates else 1.0
        else:
            growth_rates[d] = 1.0

    return {
        "tracked_dims": target_dims,
        "magnitude_trajectories": trajectories,
        "emergence_layers": emergence,
        "growth_rates": growth_rates,
    }


def outlier_removal_effect(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    dims_to_clamp: Optional[list] = None,
    layer: int = -1,
    clamp_value: float = 0.0,
) -> dict:
    """Measure the impact of clamping outlier dimensions.

    Clamps specific dimensions to a fixed value and measures how
    the output distribution changes.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        dims_to_clamp: Dimensions to clamp. Auto-detects outliers if None.
        layer: Layer to clamp at (-1 = last).
        clamp_value: Value to set clamped dimensions to.

    Returns:
        Dict with:
            "clamped_dims": list of clamped dimensions
            "kl_divergence": KL divergence between original and clamped output
            "top_token_change": change in top-1 token probability
            "prediction_changed": whether top prediction changed
            "original_entropy": entropy of original output
            "clamped_entropy": entropy of clamped output
    """
    if layer == -1:
        layer = model.cfg.n_layers - 1

    if dims_to_clamp is None:
        result = detect_outlier_dimensions(model, tokens, layer=layer)
        dims_to_clamp = [d for d, _ in result["outlier_dims"][:5]]

    # Original output
    original_logits = np.array(model(tokens)[-1])
    original_probs = np.exp(original_logits - np.max(original_logits))
    original_probs = original_probs / np.sum(original_probs)

    # Clamped output
    hook_name = f"blocks.{layer}.hook_resid_post"

    def clamp_hook(x, name):
        for d in dims_to_clamp:
            x = x.at[:, d].set(clamp_value)
        return x

    clamped_logits = np.array(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, clamp_hook)])[-1])
    clamped_probs = np.exp(clamped_logits - np.max(clamped_logits))
    clamped_probs = clamped_probs / np.sum(clamped_probs)

    # KL divergence
    kl = float(np.sum(original_probs * np.log((original_probs + 1e-10) / (clamped_probs + 1e-10))))

    # Top token change
    orig_top = int(np.argmax(original_probs))
    clamp_top = int(np.argmax(clamped_probs))
    top_change = float(abs(original_probs[orig_top] - clamped_probs[orig_top]))

    # Entropy
    orig_entropy = -float(np.sum(original_probs * np.log(original_probs + 1e-10)))
    clamp_entropy = -float(np.sum(clamped_probs * np.log(clamped_probs + 1e-10)))

    return {
        "clamped_dims": dims_to_clamp,
        "kl_divergence": kl,
        "top_token_change": top_change,
        "prediction_changed": orig_top != clamp_top,
        "original_entropy": orig_entropy,
        "clamped_entropy": clamp_entropy,
    }


def attention_sink_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Analyze attention sink tokens and their activation patterns.

    Attention sinks are tokens (often the first token or special tokens)
    that receive disproportionate attention from many heads. This analyzes
    their residual stream properties.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "attention_received": [seq_len] mean attention weight received per position
            "sink_positions": positions receiving > 2x mean attention
            "sink_activation_norms": [n_sinks] residual stream norms at sink positions
            "non_sink_mean_norm": mean residual norm at non-sink positions
            "sink_vs_nonsink_ratio": ratio of sink to non-sink activation norms
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    # Aggregate attention patterns across all heads and layers
    total_attention = np.zeros(seq_len)
    n_patterns = 0

    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key in cache.cache_dict:
            pattern = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
            # Sum attention received by each position (column sum)
            for h in range(pattern.shape[0]):
                total_attention += np.sum(pattern[h], axis=0)  # sum over query positions
                n_patterns += 1

    if n_patterns > 0:
        total_attention /= n_patterns

    # Identify sinks: positions receiving > 2x mean attention
    mean_attn = float(np.mean(total_attention))
    sink_mask = total_attention > 2.0 * mean_attn
    sink_positions = list(np.where(sink_mask)[0].astype(int))

    # Get residual stream norms at last layer
    last_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if last_key in cache.cache_dict:
        resid = np.array(cache.cache_dict[last_key])  # [seq, d_model]
        norms = np.linalg.norm(resid, axis=-1)

        sink_norms = [float(norms[p]) for p in sink_positions] if sink_positions else []
        non_sink_mask = ~sink_mask
        non_sink_mean = float(np.mean(norms[non_sink_mask])) if np.sum(non_sink_mask) > 0 else 0.0
        sink_mean = float(np.mean(sink_norms)) if sink_norms else 0.0
        ratio = sink_mean / (non_sink_mean + 1e-10)
    else:
        sink_norms = []
        non_sink_mean = 0.0
        ratio = 1.0

    return {
        "attention_received": total_attention,
        "sink_positions": sink_positions,
        "sink_activation_norms": np.array(sink_norms),
        "non_sink_mean_norm": non_sink_mean,
        "sink_vs_nonsink_ratio": ratio,
    }


def dimension_utilization_spectrum(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int = -1,
) -> dict:
    """Analyze how evenly the model uses its d_model dimensions.

    Computes the variance explained by each dimension and measures
    the concentration of representational capacity.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer to analyze (-1 = last).

    Returns:
        Dict with:
            "variance_per_dim": [d_model] sorted variance contribution per dimension
            "effective_dimensionality": number of dimensions explaining 90% of variance
            "gini_coefficient": Gini coefficient of variance distribution (0=equal, 1=concentrated)
            "top_10_fraction": fraction of variance in top 10% of dimensions
            "utilization_entropy": entropy of the variance distribution
    """
    _, cache = model.run_with_cache(tokens)

    if layer == -1:
        layer = model.cfg.n_layers - 1

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache.cache_dict:
        d = model.cfg.d_model
        return {
            "variance_per_dim": np.ones(d) / d,
            "effective_dimensionality": d,
            "gini_coefficient": 0.0,
            "top_10_fraction": 0.1,
            "utilization_entropy": float(np.log(d)),
        }

    act = np.array(cache.cache_dict[key])  # [seq, d_model]

    # Per-dimension variance
    var = np.var(act, axis=0)  # [d_model]
    var_sorted = np.sort(var)[::-1]

    total_var = np.sum(var_sorted)
    var_norm = var_sorted / (total_var + 1e-10)

    # Effective dimensionality: dims for 90% variance
    cumsum = np.cumsum(var_norm)
    eff_dim = int(np.searchsorted(cumsum, 0.9)) + 1

    # Gini coefficient
    d = len(var_sorted)
    if d > 0 and total_var > 1e-10:
        index = np.arange(1, d + 1)
        gini = float((2 * np.sum(index * var_sorted) / (d * total_var)) - (d + 1) / d)
        gini = max(0.0, min(1.0, abs(gini)))
    else:
        gini = 0.0

    # Top 10% fraction
    top_k = max(1, d // 10)
    top_frac = float(np.sum(var_sorted[:top_k]) / (total_var + 1e-10))

    # Utilization entropy
    var_safe = var_norm[var_norm > 1e-10]
    entropy = -float(np.sum(var_safe * np.log(var_safe + 1e-10)))

    return {
        "variance_per_dim": var_sorted,
        "effective_dimensionality": eff_dim,
        "gini_coefficient": gini,
        "top_10_fraction": top_frac,
        "utilization_entropy": entropy,
    }
