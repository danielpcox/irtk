"""Principled attribution via Shapley values and game-theoretic methods.

Provides theoretically grounded attribution scores that satisfy
fundamental axioms (efficiency, symmetry, null player). Unlike
gradient-based methods, these give faithfully additive importance scores.

References:
    - Lundstrom et al. (2022) "A Rigorous Study of Integrated Gradients Method
      and Extensions to Internal Neuron Attributions"
    - Chen et al. (2023) "Algorithms for the Shapley Value of Cooperative Games"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def shapley_value_tokens(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    n_samples: int = 100,
    baseline: str = "zero",
    seed: int = 42,
) -> dict:
    """Compute Shapley value importance for each input token.

    Uses Monte Carlo sampling of permutations to estimate Shapley values.
    Each token's value is its average marginal contribution across orderings.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        metric_fn: Function from logits -> float.
        n_samples: Number of permutation samples (more = more accurate).
        baseline: How to replace ablated tokens ("zero" or "mean").
        seed: Random seed.

    Returns:
        Dict with:
        - "shapley_values": [seq_len] attribution score per token
        - "null_value": metric with all tokens ablated
        - "full_value": metric with all tokens present
        - "efficiency_gap": |sum(shapley) - (full - null)| (should be ~0)
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)
    rng = np.random.RandomState(seed)

    # Baseline embedding
    _, cache = model.run_with_cache(tokens)
    clean_embed = np.array(cache.cache_dict.get("hook_embed", np.zeros((seq_len, model.cfg.d_model))))

    if baseline == "zero":
        baseline_embed = np.zeros_like(clean_embed)
    else:
        baseline_embed = np.full_like(clean_embed, np.mean(clean_embed))

    # Full and null values
    full_logits = model(tokens)
    full_value = float(metric_fn(full_logits))

    def make_mask_hook(mask, clean_e, baseline_e):
        mixed = jnp.array(mask[:, None] * clean_e + (1 - mask[:, None]) * baseline_e)
        def hook(x, name):
            return mixed
        return hook

    null_mask = np.zeros(seq_len)
    null_logits = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", make_mask_hook(null_mask, clean_embed, baseline_embed))]
    )
    null_value = float(metric_fn(null_logits))

    # Monte Carlo Shapley estimation
    shapley = np.zeros(seq_len)

    for _ in range(n_samples):
        perm = rng.permutation(seq_len)
        mask = np.zeros(seq_len)
        prev_value = null_value

        for idx in perm:
            mask[idx] = 1.0
            logits = model.run_with_hooks(
                tokens, fwd_hooks=[("hook_embed", make_mask_hook(mask, clean_embed, baseline_embed))]
            )
            curr_value = float(metric_fn(logits))
            shapley[idx] += curr_value - prev_value
            prev_value = curr_value

    shapley /= n_samples
    efficiency_gap = abs(float(np.sum(shapley)) - (full_value - null_value))

    return {
        "shapley_values": shapley,
        "null_value": null_value,
        "full_value": full_value,
        "efficiency_gap": efficiency_gap,
    }


def kernel_shap_activations(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    metric_fn: Callable,
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Fast SHAP approximation for activation dimension contributions.

    Uses KernelSHAP (weighted linear regression) to estimate importance
    of each activation dimension at a given hook point.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        hook_name: Hook point to analyze.
        metric_fn: Function from logits -> float.
        n_samples: Number of random masks to sample.
        seed: Random seed.

    Returns:
        Dict with:
        - "dimension_importance": [d_model] importance per dimension
        - "top_dimensions": list of (dim_idx, importance) top-10
        - "r_squared": fit quality of the linear approximation
    """
    tokens = jnp.array(tokens)
    rng = np.random.RandomState(seed)

    _, cache = model.run_with_cache(tokens)
    if hook_name not in cache.cache_dict:
        d = model.cfg.d_model
        return {"dimension_importance": np.zeros(d), "top_dimensions": [], "r_squared": 0.0}

    clean_act = np.array(cache.cache_dict[hook_name])  # [seq, d_model]
    d_model = clean_act.shape[-1]

    # Sample binary masks over dimensions
    masks = []
    values = []

    for _ in range(n_samples):
        mask = rng.binomial(1, 0.5, size=d_model).astype(float)
        masks.append(mask)

        masked_act = jnp.array(clean_act * mask[None, :])

        def make_hook(ma):
            def hook(x, name):
                return ma
            return hook

        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(masked_act))])
        values.append(float(metric_fn(logits)))

    Z = np.array(masks)  # [n_samples, d_model]
    y = np.array(values)  # [n_samples]

    # Weighted linear regression (KernelSHAP weights)
    n_active = Z.sum(axis=1)
    weights = np.ones(n_samples)
    for i in range(n_samples):
        k = n_active[i]
        if 0 < k < d_model:
            from math import comb
            denom = comb(d_model, int(k))
            if denom > 0:
                weights[i] = (d_model - 1) / (denom * k * (d_model - k))

    # Solve weighted least squares: W @ Z @ beta = W @ y
    W = np.diag(np.sqrt(weights))
    Zw = W @ Z
    yw = W @ y

    # Ridge regression for stability
    lam = 1e-6
    beta = np.linalg.solve(Zw.T @ Zw + lam * np.eye(d_model), Zw.T @ yw)

    # R-squared
    y_pred = Z @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    top_idx = np.argsort(np.abs(beta))[::-1][:10]
    top_dims = [(int(i), float(beta[i])) for i in top_idx]

    return {
        "dimension_importance": beta,
        "top_dimensions": top_dims,
        "r_squared": float(r2),
    }


def interaction_index(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    components: list[str],
    n_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Compute pairwise interaction indices between model components.

    The interaction index measures whether two components have synergistic
    (positive) or redundant (negative) effects beyond their individual
    contributions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        metric_fn: Function from logits -> float.
        components: List of hook names to test interactions between.
        n_samples: Samples for MC estimation.
        seed: Random seed.

    Returns:
        Dict with:
        - "interaction_matrix": [n_comp, n_comp] pairwise interaction strengths
        - "individual_effects": [n_comp] individual marginal effects
        - "most_synergistic": (comp_i, comp_j, value) strongest positive interaction
        - "most_redundant": (comp_i, comp_j, value) strongest negative interaction
    """
    tokens = jnp.array(tokens)
    n_comp = len(components)
    rng = np.random.RandomState(seed)

    # Full metric
    full_logits = model(tokens)
    full_value = float(metric_fn(full_logits))

    # Individual effects (zero ablation)
    individual = np.zeros(n_comp)
    for i, comp in enumerate(components):
        def zero_hook(x, name):
            return jnp.zeros_like(x)
        logits = model.run_with_hooks(tokens, fwd_hooks=[(comp, zero_hook)])
        individual[i] = full_value - float(metric_fn(logits))

    # Pairwise interaction: I(i,j) = effect_joint - effect_i - effect_j
    interaction = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            def zero_both(x, name):
                return jnp.zeros_like(x)

            logits = model.run_with_hooks(
                tokens, fwd_hooks=[
                    (components[i], zero_both),
                    (components[j], zero_both),
                ]
            )
            joint_effect = full_value - float(metric_fn(logits))
            inter = joint_effect - individual[i] - individual[j]
            interaction[i, j] = inter
            interaction[j, i] = inter

    # Find extremes
    triu = np.triu(interaction, k=1)
    max_idx = np.unravel_index(np.argmax(triu), triu.shape)
    min_idx = np.unravel_index(np.argmin(triu), triu.shape)

    return {
        "interaction_matrix": interaction,
        "individual_effects": individual,
        "most_synergistic": (
            components[max_idx[0]], components[max_idx[1]],
            float(interaction[max_idx])
        ),
        "most_redundant": (
            components[min_idx[0]], components[min_idx[1]],
            float(interaction[min_idx])
        ),
    }


def path_attribution_value(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Trace layer-by-layer attribution showing each layer's contribution.

    Computes the marginal effect of each layer by ablating it and measuring
    the metric change. Provides a layer-wise decomposition of the model's
    computation.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "layer_attributions": [n_layers] contribution per layer
        - "attn_attributions": [n_layers] attention contribution per layer
        - "mlp_attributions": [n_layers] MLP contribution per layer
        - "residual_fraction": fraction of value from residual (non-attributed)
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers

    full_logits = model(tokens)
    full_value = float(metric_fn(full_logits))

    attn_attr = np.zeros(n_layers)
    mlp_attr = np.zeros(n_layers)

    for layer in range(n_layers):
        # Attn ablation
        attn_hook = f"blocks.{layer}.hook_attn_out"

        def zero_attn(x, name):
            return jnp.zeros_like(x)

        logits = model.run_with_hooks(tokens, fwd_hooks=[(attn_hook, zero_attn)])
        attn_attr[layer] = full_value - float(metric_fn(logits))

        # MLP ablation
        mlp_hook = f"blocks.{layer}.hook_mlp_out"

        def zero_mlp(x, name):
            return jnp.zeros_like(x)

        logits = model.run_with_hooks(tokens, fwd_hooks=[(mlp_hook, zero_mlp)])
        mlp_attr[layer] = full_value - float(metric_fn(logits))

    layer_attr = attn_attr + mlp_attr
    total_attributed = float(np.sum(layer_attr))
    residual_fraction = 1.0 - total_attributed / max(abs(full_value), 1e-10)

    return {
        "layer_attributions": layer_attr,
        "attn_attributions": attn_attr,
        "mlp_attributions": mlp_attr,
        "residual_fraction": float(residual_fraction),
    }


def importance_ranking_with_std(
    model: HookedTransformer,
    tokens_list: list,
    metric_fn: Callable,
    components: list[str],
) -> dict:
    """Rank components by importance with confidence intervals across prompts.

    Computes per-component ablation effects across multiple prompts,
    providing mean importance and standard deviation for significance.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        metric_fn: Function from logits -> float.
        components: List of hook names to rank.

    Returns:
        Dict with:
        - "ranking": list of (component, mean_effect, std_effect) sorted by |mean|
        - "effect_matrix": [n_components, n_prompts] raw effects
        - "significant_components": components where |mean| > 2*std
    """
    n_comp = len(components)
    n_prompts = len(tokens_list)

    effects = np.zeros((n_comp, n_prompts))

    for pi, tokens in enumerate(tokens_list):
        tokens = jnp.array(tokens)
        clean_logits = model(tokens)
        clean_value = float(metric_fn(clean_logits))

        for ci, comp in enumerate(components):
            def zero_hook(x, name):
                return jnp.zeros_like(x)

            logits = model.run_with_hooks(tokens, fwd_hooks=[(comp, zero_hook)])
            effects[ci, pi] = clean_value - float(metric_fn(logits))

    means = np.mean(effects, axis=1)
    stds = np.std(effects, axis=1)

    # Rank by absolute mean effect
    order = np.argsort(np.abs(means))[::-1]
    ranking = [(components[i], float(means[i]), float(stds[i])) for i in order]

    # Significant: |mean| > 2*std (rough significance criterion)
    significant = [
        components[i] for i in range(n_comp)
        if abs(means[i]) > 2 * max(stds[i], 1e-10)
    ]

    return {
        "ranking": ranking,
        "effect_matrix": effects,
        "significant_components": significant,
    }
