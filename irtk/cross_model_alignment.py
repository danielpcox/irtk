"""Cross-model structural alignment.

Aligns circuits, attention heads, and features across different model
families — enabling universality quantification and comparative analysis
beyond simple weight/activation diffs.

Functions:
- cka_layer_correspondence: CKA similarity between layers of two models
- match_heads_across_models: Find matching attention heads across models
- circuit_universality_score: How universal is a circuit across models
- aligned_feature_comparison: Match SAE features across models
- scale_law_trajectory: Track a circuit metric across model scales

References:
    - Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
    - Elhage et al. (2022) "A Mathematical Framework for Transformer Circuits"
    - Chughtai et al. (2023) "A Toy Model of Universality"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


def _linear_cka(X, Y):
    """Compute linear CKA between two representation matrices."""
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    # Center
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    # HSIC with linear kernel
    XtX = X @ X.T
    YtY = Y @ Y.T

    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def cka_layer_correspondence(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
    hook_names_a: Optional[list] = None,
    hook_names_b: Optional[list] = None,
) -> dict:
    """Compute CKA between all pairs of layers from two models.

    Args:
        model_a: First HookedTransformer.
        model_b: Second HookedTransformer.
        tokens: [seq_len] input tokens (must be valid for both models).
        hook_names_a: Hook names for model_a. Defaults to all resid_post.
        hook_names_b: Hook names for model_b. Defaults to all resid_post.

    Returns:
        Dict with:
            "cka_matrix": [len(hooks_a), len(hooks_b)] CKA scores
            "best_match": dict mapping each layer_a to best-matching layer_b
            "mean_cka": mean CKA across all pairs
    """
    _, cache_a = model_a.run_with_cache(tokens)
    _, cache_b = model_b.run_with_cache(tokens)

    if hook_names_a is None:
        hook_names_a = [f"blocks.{l}.hook_resid_post" for l in range(model_a.cfg.n_layers)]
    if hook_names_b is None:
        hook_names_b = [f"blocks.{l}.hook_resid_post" for l in range(model_b.cfg.n_layers)]

    n_a = len(hook_names_a)
    n_b = len(hook_names_b)
    cka_matrix = np.zeros((n_a, n_b))

    for i, ha in enumerate(hook_names_a):
        act_a = cache_a.cache_dict.get(ha)
        if act_a is None:
            continue
        X = np.array(act_a)  # [seq_len, d_model_a]

        for j, hb in enumerate(hook_names_b):
            act_b = cache_b.cache_dict.get(hb)
            if act_b is None:
                continue
            Y = np.array(act_b)  # [seq_len, d_model_b]
            cka_matrix[i, j] = _linear_cka(X, Y)

    # Best matches
    best_match = {}
    for i in range(n_a):
        best_j = int(np.argmax(cka_matrix[i]))
        best_match[hook_names_a[i]] = hook_names_b[best_j]

    return {
        "cka_matrix": cka_matrix,
        "best_match": best_match,
        "mean_cka": float(np.mean(cka_matrix)),
    }


def match_heads_across_models(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
    metric: str = "pattern_correlation",
) -> dict:
    """Find matching attention heads across two models.

    Args:
        model_a: First HookedTransformer.
        model_b: Second HookedTransformer.
        tokens: [seq_len] input tokens.
        metric: Similarity metric. One of:
            "pattern_correlation" - correlation of attention patterns
            "ov_cosine" - cosine similarity of OV circuit directions

    Returns:
        Dict with:
            "matches": list of ((layer_a, head_a), (layer_b, head_b), score)
            "similarity_matrix": [total_heads_a, total_heads_b]
            "best_match_per_head_a": dict (layer, head) -> (layer, head) in model_b
    """
    _, cache_a = model_a.run_with_cache(tokens)
    _, cache_b = model_b.run_with_cache(tokens)

    n_layers_a, n_heads_a = model_a.cfg.n_layers, model_a.cfg.n_heads
    n_layers_b, n_heads_b = model_b.cfg.n_layers, model_b.cfg.n_heads
    total_a = n_layers_a * n_heads_a
    total_b = n_layers_b * n_heads_b

    labels_a = [(l, h) for l in range(n_layers_a) for h in range(n_heads_a)]
    labels_b = [(l, h) for l in range(n_layers_b) for h in range(n_heads_b)]

    sim_matrix = np.zeros((total_a, total_b))

    if metric == "pattern_correlation":
        for i, (la, ha) in enumerate(labels_a):
            pattern_key_a = f"blocks.{la}.attn.hook_pattern"
            if pattern_key_a not in cache_a.cache_dict:
                continue
            pat_a = np.array(cache_a.cache_dict[pattern_key_a][:, ha, :]).flatten()

            for j, (lb, hb) in enumerate(labels_b):
                pattern_key_b = f"blocks.{lb}.attn.hook_pattern"
                if pattern_key_b not in cache_b.cache_dict:
                    continue
                pat_b = np.array(cache_b.cache_dict[pattern_key_b][:, hb, :]).flatten()

                # Pad to same length if needed
                min_len = min(len(pat_a), len(pat_b))
                if min_len > 0:
                    corr = np.corrcoef(pat_a[:min_len], pat_b[:min_len])[0, 1]
                    sim_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    elif metric == "ov_cosine":
        for i, (la, ha) in enumerate(labels_a):
            z_key_a = f"blocks.{la}.attn.hook_z"
            if z_key_a not in cache_a.cache_dict:
                continue
            z_a = np.array(cache_a.cache_dict[z_key_a][:, ha, :]).flatten()
            norm_a = np.linalg.norm(z_a)

            for j, (lb, hb) in enumerate(labels_b):
                z_key_b = f"blocks.{lb}.attn.hook_z"
                if z_key_b not in cache_b.cache_dict:
                    continue
                z_b = np.array(cache_b.cache_dict[z_key_b][:, hb, :]).flatten()
                norm_b = np.linalg.norm(z_b)

                min_len = min(len(z_a), len(z_b))
                if min_len > 0 and norm_a > 1e-10 and norm_b > 1e-10:
                    sim_matrix[i, j] = float(
                        np.dot(z_a[:min_len], z_b[:min_len]) /
                        (np.linalg.norm(z_a[:min_len]) * np.linalg.norm(z_b[:min_len]) + 1e-10)
                    )

    # Find best matches
    matches = []
    best_per_head = {}
    for i, (la, ha) in enumerate(labels_a):
        best_j = int(np.argmax(np.abs(sim_matrix[i])))
        lb, hb = labels_b[best_j]
        score = float(sim_matrix[i, best_j])
        matches.append(((la, ha), (lb, hb), score))
        best_per_head[(la, ha)] = (lb, hb)

    matches.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "matches": matches,
        "similarity_matrix": sim_matrix,
        "best_match_per_head_a": best_per_head,
    }


def circuit_universality_score(
    circuit_heads: list,
    models: list,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Measure how universal a circuit is across multiple models.

    Runs the same ablation experiment on each model and measures variance
    in the faithfulness score. Low variance = universal circuit.

    Args:
        circuit_heads: List of (layer, head) tuples defining the circuit.
        models: List of HookedTransformers to test.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "per_model_faithfulness": list of faithfulness scores
            "mean_faithfulness": mean across models
            "std_faithfulness": standard deviation
            "universality_score": 1 - normalized_std (higher = more universal)
    """
    if not models or not circuit_heads:
        return {
            "per_model_faithfulness": [],
            "mean_faithfulness": 0.0,
            "std_faithfulness": 0.0,
            "universality_score": 0.0,
        }

    faithfulness_scores = []

    for model in models:
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Clean metric
        clean_logits = model(tokens)
        clean_metric = float(metric_fn(clean_logits))

        # Ablate everything except circuit
        all_heads = {(l, h) for l in range(n_layers) for h in range(n_heads)}
        circuit_set = set((l, h) for l, h in circuit_heads
                         if l < n_layers and h < n_heads)
        non_circuit = all_heads - circuit_set

        hooks = []
        for l, h in non_circuit:
            hook_name = f"blocks.{l}.attn.hook_z"

            def make_hook(_h=h):
                def hook_fn(x, name):
                    return x.at[:, _h, :].set(0.0)
                return hook_fn

            hooks.append((hook_name, make_hook()))

        if hooks:
            circuit_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            circuit_logits = model(tokens)
        circuit_metric = float(metric_fn(circuit_logits))

        # Faithfulness: how much of the metric does the circuit preserve?
        if abs(clean_metric) > 1e-8:
            faith = circuit_metric / clean_metric
        else:
            faith = 1.0 if abs(circuit_metric) < 1e-8 else 0.0

        faithfulness_scores.append(faith)

    mean_f = float(np.mean(faithfulness_scores))
    std_f = float(np.std(faithfulness_scores))
    # Universality: 1 - normalized std
    universality = 1.0 - min(std_f / (abs(mean_f) + 1e-8), 1.0)

    return {
        "per_model_faithfulness": faithfulness_scores,
        "mean_faithfulness": mean_f,
        "std_faithfulness": std_f,
        "universality_score": universality,
    }


def aligned_feature_comparison(
    sae_a: SparseAutoencoder,
    sae_b: SparseAutoencoder,
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name_a: str,
    hook_name_b: str,
    top_k: int = 10,
) -> dict:
    """Find maximally similar SAE features between two models.

    Computes cosine similarity between decoder directions and matches
    features with correlated activation patterns.

    Args:
        sae_a: SAE trained on model_a activations.
        sae_b: SAE trained on model_b activations.
        model_a: First HookedTransformer.
        model_b: Second HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name_a: Hook point for model_a.
        hook_name_b: Hook point for model_b.
        top_k: Number of top matches to return.

    Returns:
        Dict with:
            "matched_pairs": list of (feature_a, feature_b, score)
            "activation_correlation": correlation of activation patterns
            "mean_similarity": mean cosine similarity of matches
    """
    # Get activations
    _, cache_a = model_a.run_with_cache(tokens)
    _, cache_b = model_b.run_with_cache(tokens)

    act_a = cache_a.cache_dict.get(hook_name_a)
    act_b = cache_b.cache_dict.get(hook_name_b)

    if act_a is None or act_b is None:
        return {
            "matched_pairs": [],
            "activation_correlation": 0.0,
            "mean_similarity": 0.0,
        }

    # Feature activations
    feat_acts_a = np.array(sae_a.encode(act_a))  # [seq, n_features_a]
    feat_acts_b = np.array(sae_b.encode(act_b))  # [seq, n_features_b]

    # Correlation of activation patterns
    n_a = feat_acts_a.shape[1]
    n_b = feat_acts_b.shape[1]

    # Compute cosine similarity between mean activations
    mean_a = np.mean(feat_acts_a, axis=0)  # [n_features_a]
    mean_b = np.mean(feat_acts_b, axis=0)  # [n_features_b]

    # Normalize
    norm_a = mean_a / (np.linalg.norm(mean_a) + 1e-10)
    norm_b = mean_b / (np.linalg.norm(mean_b) + 1e-10)

    # For each feature in A, find best match in B based on activation correlation
    matched_pairs = []
    seq_len = min(feat_acts_a.shape[0], feat_acts_b.shape[0])

    for i in range(min(n_a, 50)):  # Limit computation
        act_i = feat_acts_a[:seq_len, i]
        if np.std(act_i) < 1e-8:
            continue
        best_score = -1.0
        best_j = 0
        for j in range(min(n_b, 50)):
            act_j = feat_acts_b[:seq_len, j]
            if np.std(act_j) < 1e-8:
                continue
            corr = np.corrcoef(act_i, act_j)[0, 1]
            if not np.isnan(corr) and abs(corr) > best_score:
                best_score = abs(corr)
                best_j = j
        if best_score > 0:
            matched_pairs.append((i, best_j, best_score))

    matched_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = matched_pairs[:top_k]

    mean_sim = float(np.mean([s for _, _, s in top_pairs])) if top_pairs else 0.0

    # Overall activation correlation
    flat_a = feat_acts_a[:seq_len].flatten()
    flat_b = feat_acts_b[:seq_len].flatten()
    min_len = min(len(flat_a), len(flat_b))
    if min_len > 0 and np.std(flat_a[:min_len]) > 1e-8 and np.std(flat_b[:min_len]) > 1e-8:
        overall_corr = float(np.corrcoef(flat_a[:min_len], flat_b[:min_len])[0, 1])
        if np.isnan(overall_corr):
            overall_corr = 0.0
    else:
        overall_corr = 0.0

    return {
        "matched_pairs": top_pairs,
        "activation_correlation": overall_corr,
        "mean_similarity": mean_sim,
    }


def scale_law_trajectory(
    models: list,
    model_sizes: list,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    circuit_heads: Optional[list] = None,
) -> dict:
    """Track a metric as a function of model parameter count.

    Args:
        models: List of HookedTransformers of increasing size.
        model_sizes: List of model sizes (e.g., parameter counts).
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        circuit_heads: Optional list of (layer, head) for circuit-specific metrics.

    Returns:
        Dict with:
            "sizes": sorted model sizes
            "metrics": corresponding metric values
            "trend": "increasing", "decreasing", or "non_monotonic"
            "log_log_slope": slope of log-log fit (scaling exponent)
    """
    if not models or not model_sizes:
        return {
            "sizes": [],
            "metrics": [],
            "trend": "flat",
            "log_log_slope": 0.0,
        }

    # Sort by size
    pairs = sorted(zip(model_sizes, models), key=lambda x: x[0])
    sorted_sizes = [s for s, _ in pairs]
    sorted_models = [m for _, m in pairs]

    metrics = []
    for model in sorted_models:
        logits = model(tokens)
        metrics.append(float(metric_fn(logits)))

    # Trend
    if len(metrics) >= 2:
        diffs = np.diff(metrics)
        if np.all(diffs >= -0.01):
            trend = "increasing"
        elif np.all(diffs <= 0.01):
            trend = "decreasing"
        else:
            trend = "non_monotonic"
    else:
        trend = "flat"

    # Log-log slope (scaling exponent)
    if len(sorted_sizes) >= 2:
        log_sizes = np.log(np.array(sorted_sizes, dtype=np.float64) + 1)
        log_metrics = np.log(np.abs(np.array(metrics, dtype=np.float64)) + 1e-10)
        if np.std(log_sizes) > 1e-8:
            slope = float(np.polyfit(log_sizes, log_metrics, 1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0

    return {
        "sizes": sorted_sizes,
        "metrics": metrics,
        "trend": trend,
        "log_log_slope": slope,
    }
