"""LayerNorm mechanics analysis.

Investigates LayerNorm's role in feature dynamics — how it scales, suppresses,
and selects feature directions. LayerNorm is often treated as a black box,
but its learned weights and centering operation have significant effects on
which features survive and how they're weighted.

Functions:
- feature_scaling_by_norm: How LayerNorm scales each feature direction
- norm_directionality_bias: Which directions are amplified/suppressed by LN weights
- pre_vs_post_norm_effect: Compare representations before and after normalization
- norm_gradient_flow: Gradient magnitude through LayerNorm for each feature
- feature_whitening_by_layer: Decorrelation induced by normalization

References:
    - Brody et al. (2023) "On the Expressivity Role of LayerNorm in Transformers"
    - Xu et al. (2019) "Understanding and Improving Layer Normalization"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def feature_scaling_by_norm(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
) -> dict:
    """Measure how LayerNorm at a given layer scales each feature direction.

    Compares the norm of each dimension before and after LayerNorm to
    understand which features are amplified vs suppressed.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index to analyze.

    Returns:
        Dict with:
            "scaling_factors": [d_model] per-dimension scaling factors
            "most_amplified": index of most amplified dimension
            "most_suppressed": index of most suppressed dimension
            "mean_scaling": mean scaling factor across dimensions
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f"blocks.{layer}.hook_resid_pre"
    # After LN but before attention/MLP
    ln_scale_key = f"blocks.{layer}.ln1.hook_scale"
    ln_norm_key = f"blocks.{layer}.ln1.hook_normalized"

    pre_act = cache.cache_dict.get(pre_key)
    normalized = cache.cache_dict.get(ln_norm_key)

    if pre_act is None or normalized is None:
        d = model.cfg.d_model
        return {
            "scaling_factors": np.ones(d),
            "most_amplified": 0,
            "most_suppressed": 0,
            "mean_scaling": 1.0,
        }

    pre_act = np.array(pre_act)
    normalized = np.array(normalized)

    # Per-dimension variance before and after
    pre_var = np.var(pre_act, axis=0)  # [d_model]
    post_var = np.var(normalized, axis=0)  # [d_model]

    # Scaling = sqrt(post_var / pre_var)
    scaling = np.sqrt((post_var + 1e-10) / (pre_var + 1e-10))

    # Apply LN weights if available
    block = model.blocks[layer]
    if hasattr(block, 'ln1') and hasattr(block.ln1, 'w'):
        w = np.array(block.ln1.w)
        scaling = scaling * np.abs(w)

    return {
        "scaling_factors": scaling,
        "most_amplified": int(np.argmax(scaling)),
        "most_suppressed": int(np.argmin(scaling)),
        "mean_scaling": float(np.mean(scaling)),
    }


def norm_directionality_bias(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
) -> dict:
    """Identify which feature directions are biased by LayerNorm weights.

    LayerNorm weights (w) and bias (b) create a directional preference
    in the residual stream. This function quantifies that bias.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index to analyze.

    Returns:
        Dict with:
            "weight_direction": [d_model] normalized LN weight vector
            "bias_direction": [d_model] LN bias (if available)
            "weight_magnitude_range": (min, max) of |w|
            "anisotropy": ratio of max/min weight magnitudes
    """
    block = model.blocks[layer]

    if hasattr(block, 'ln1') and hasattr(block.ln1, 'w'):
        w = np.array(block.ln1.w)
    else:
        w = np.ones(model.cfg.d_model)

    if hasattr(block, 'ln1') and hasattr(block.ln1, 'b'):
        b = np.array(block.ln1.b)
    else:
        b = np.zeros(model.cfg.d_model)

    w_abs = np.abs(w)
    w_norm = w / (np.linalg.norm(w) + 1e-10)

    min_w = float(np.min(w_abs))
    max_w = float(np.max(w_abs))
    anisotropy = max_w / (min_w + 1e-10)

    return {
        "weight_direction": w_norm,
        "bias_direction": b,
        "weight_magnitude_range": (min_w, max_w),
        "anisotropy": anisotropy,
    }


def pre_vs_post_norm_effect(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
) -> dict:
    """Compare representations before and after LayerNorm.

    Quantifies how much LayerNorm changes the representation geometry:
    cosine similarity, rank change, and information content.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index to analyze.

    Returns:
        Dict with:
            "cosine_similarity": mean cosine similarity pre vs post per token
            "norm_ratio": ratio of post/pre norms
            "rank_pre": effective rank before LN
            "rank_post": effective rank after LN
            "centering_effect": magnitude of the mean subtraction
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f"blocks.{layer}.hook_resid_pre"
    norm_key = f"blocks.{layer}.ln1.hook_normalized"

    pre = cache.cache_dict.get(pre_key)
    post = cache.cache_dict.get(norm_key)

    if pre is None or post is None:
        return {
            "cosine_similarity": 0.0,
            "norm_ratio": 1.0,
            "rank_pre": 0,
            "rank_post": 0,
            "centering_effect": 0.0,
        }

    pre = np.array(pre)  # [seq, d_model]
    post = np.array(post)

    # Apply LN weights if present
    block = model.blocks[layer]
    if hasattr(block, 'ln1') and hasattr(block.ln1, 'w'):
        w = np.array(block.ln1.w)
        b = np.array(getattr(block.ln1, 'b', np.zeros_like(w)))
        post_weighted = post * w + b
    else:
        post_weighted = post

    # Per-token cosine similarity
    cos_sims = []
    for i in range(pre.shape[0]):
        cs = np.dot(pre[i], post_weighted[i]) / (
            np.linalg.norm(pre[i]) * np.linalg.norm(post_weighted[i]) + 1e-10
        )
        cos_sims.append(cs)

    # Norm ratio
    pre_norms = np.linalg.norm(pre, axis=-1)
    post_norms = np.linalg.norm(post_weighted, axis=-1)
    norm_ratio = float(np.mean(post_norms / (pre_norms + 1e-10)))

    # Effective rank via SVD
    def eff_rank(X):
        s = np.linalg.svd(X, compute_uv=False)
        s = s / (np.sum(s) + 1e-10)
        s = s[s > 1e-10]
        return float(np.exp(-np.sum(s * np.log(s + 1e-10))))

    rank_pre = eff_rank(pre)
    rank_post = eff_rank(post_weighted)

    # Centering effect
    centering = float(np.mean(np.linalg.norm(
        pre - np.mean(pre, axis=-1, keepdims=True) - pre, axis=-1
    )))

    return {
        "cosine_similarity": float(np.mean(cos_sims)),
        "norm_ratio": norm_ratio,
        "rank_pre": rank_pre,
        "rank_post": rank_post,
        "centering_effect": centering,
    }


def norm_gradient_flow(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    metric_fn=None,
) -> dict:
    """Analyze gradient magnitude through LayerNorm for each feature dimension.

    Measures how LayerNorm affects gradient flow by comparing gradient norms
    at the input vs output of normalization.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index to analyze.
        metric_fn: Optional metric function. Defaults to logit[last, 0].

    Returns:
        Dict with:
            "pre_norm_grad_norms": [d_model] gradient norms before LN
            "post_norm_grad_norms": [d_model] gradient norms after LN
            "gradient_amplification": [d_model] ratio of post/pre gradient norms
            "mean_amplification": mean gradient amplification
    """
    d_model = model.cfg.d_model

    # Collect pre-norm activations and compute gradients
    _, cache = model.run_with_cache(tokens)
    pre_key = f"blocks.{layer}.hook_resid_pre"

    if pre_key not in cache.cache_dict:
        return {
            "pre_norm_grad_norms": np.zeros(d_model),
            "post_norm_grad_norms": np.zeros(d_model),
            "gradient_amplification": np.ones(d_model),
            "mean_amplification": 1.0,
        }

    pre_act = np.array(cache.cache_dict[pre_key])  # [seq, d_model]

    # Per-dimension gradient approximation via finite differences
    eps = 1e-4
    logits_base = model(tokens)
    base_val = float(logits_base[-1, 0]) if metric_fn is None else float(metric_fn(logits_base))

    pre_grads = np.zeros(d_model)
    for d in range(d_model):
        def perturb_hook(x, name, _d=d):
            return x.at[:, _d].add(eps)

        perturbed_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(pre_key, perturb_hook)]
        )
        perturbed_val = float(perturbed_logits[-1, 0]) if metric_fn is None else float(metric_fn(perturbed_logits))
        pre_grads[d] = abs(perturbed_val - base_val) / eps

    # Post-norm gradients (perturb after LN)
    norm_key = f"blocks.{layer}.ln1.hook_normalized"
    post_grads = np.zeros(d_model)

    if norm_key in cache.cache_dict:
        for d in range(d_model):
            def perturb_post_hook(x, name, _d=d):
                return x.at[:, _d].add(eps)

            perturbed_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(norm_key, perturb_post_hook)]
            )
            perturbed_val = float(perturbed_logits[-1, 0]) if metric_fn is None else float(metric_fn(perturbed_logits))
            post_grads[d] = abs(perturbed_val - base_val) / eps

    amplification = post_grads / (pre_grads + 1e-10)

    return {
        "pre_norm_grad_norms": pre_grads,
        "post_norm_grad_norms": post_grads,
        "gradient_amplification": amplification,
        "mean_amplification": float(np.mean(amplification)),
    }


def feature_whitening_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Measure decorrelation induced by LayerNorm at each layer.

    Computes the condition number and off-diagonal correlation magnitude
    before and after LN to quantify whitening.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "pre_correlations": [n_layers] mean off-diagonal |correlation| before LN
            "post_correlations": [n_layers] mean off-diagonal |correlation| after LN
            "whitening_effect": [n_layers] reduction in correlation (pre - post)
            "most_whitened_layer": layer with strongest decorrelation
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    pre_corrs = []
    post_corrs = []

    for l in range(n_layers):
        pre_key = f"blocks.{l}.hook_resid_pre"
        norm_key = f"blocks.{l}.ln1.hook_normalized"

        pre = cache.cache_dict.get(pre_key)
        post = cache.cache_dict.get(norm_key)

        if pre is None or post is None:
            pre_corrs.append(0.0)
            post_corrs.append(0.0)
            continue

        pre = np.array(pre)
        post = np.array(post)

        def mean_off_diag_corr(X):
            if X.shape[0] < 2:
                return 0.0
            corr = np.corrcoef(X.T)
            if np.any(np.isnan(corr)):
                return 0.0
            n = corr.shape[0]
            mask = ~np.eye(n, dtype=bool)
            return float(np.mean(np.abs(corr[mask])))

        pre_corrs.append(mean_off_diag_corr(pre))
        post_corrs.append(mean_off_diag_corr(post))

    pre_corrs = np.array(pre_corrs)
    post_corrs = np.array(post_corrs)
    whitening = pre_corrs - post_corrs

    most_whitened = int(np.argmax(whitening)) if len(whitening) > 0 else 0

    return {
        "pre_correlations": pre_corrs,
        "post_correlations": post_corrs,
        "whitening_effect": whitening,
        "most_whitened_layer": most_whitened,
    }
