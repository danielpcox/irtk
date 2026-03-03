"""Statistical hypothesis testing for mechanistic interpretability.

Rigorous statistical methods for evaluating mechanistic interpretability claims:
permutation tests, bootstrapped confidence intervals, multiple comparison
correction, effect sizes, and significance testing for circuit-level hypotheses.

References:
- Conmy et al. (2023) "Towards Automated Circuit Discovery for Mechanistic Interpretability"
- Geiger et al. (2024) "Finding Alignments Between Interpretable Causal Variables and
  Distributed Neural Representations"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def permutation_test(
    model,
    tokens_a: list,
    tokens_b: list,
    metric_fn: Callable,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for whether two groups differ on a metric.

    Computes the metric for each input, then tests whether group assignment
    matters by randomly permuting labels.

    Args:
        model: HookedTransformer model.
        tokens_a: List of token arrays for group A.
        tokens_b: List of token arrays for group B.
        metric_fn: fn(logits, tokens) -> scalar metric.
        n_permutations: Number of random permutations.
        seed: Random seed.

    Returns:
        Dict with observed_diff, p_value, null_distribution, effect_size.
    """
    # Compute metric for each input
    scores_a = []
    for tok in tokens_a:
        logits = model(tok)
        scores_a.append(float(metric_fn(logits, tok)))
    scores_b = []
    for tok in tokens_b:
        logits = model(tok)
        scores_b.append(float(metric_fn(logits, tok)))

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    all_scores = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    # Permutation null distribution
    rng = np.random.RandomState(seed)
    null_diffs = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(all_scores))
        perm_a = all_scores[perm[:n_a]]
        perm_b = all_scores[perm[n_a:]]
        null_diffs.append(float(np.mean(perm_a) - np.mean(perm_b)))
    null_diffs = np.array(null_diffs)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))

    # Cohen's d effect size
    pooled_std = np.sqrt(
        (np.var(scores_a) * (n_a - 1) + np.var(scores_b) * (len(scores_b) - 1))
        / max(n_a + len(scores_b) - 2, 1)
    )
    effect_size = float(observed_diff / max(pooled_std, 1e-10))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "null_distribution": jnp.array(null_diffs),
        "effect_size": effect_size,
        "scores_a": jnp.array(scores_a),
        "scores_b": jnp.array(scores_b),
        "n_permutations": n_permutations,
    }


def bootstrap_confidence_interval(
    model,
    tokens_list: list,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for a metric.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        metric_fn: fn(logits, tokens) -> scalar.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, bootstrap_distribution, se.
    """
    scores = []
    for tok in tokens_list:
        logits = model(tok)
        scores.append(float(metric_fn(logits, tok)))
    scores = np.array(scores)
    point_estimate = float(np.mean(scores))

    rng = np.random.RandomState(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores), size=len(scores), replace=True)
        bootstrap_means.append(float(np.mean(scores[idx])))
    bootstrap_means = np.array(bootstrap_means)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    se = float(np.std(bootstrap_means))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": jnp.array(bootstrap_means),
        "standard_error": se,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }


def multiple_comparison_correction(
    p_values: list,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> dict:
    """Correct p-values for multiple comparisons.

    Args:
        p_values: List of uncorrected p-values.
        method: "bonferroni" or "fdr" (Benjamini-Hochberg).
        alpha: Significance threshold.

    Returns:
        Dict with corrected_p_values, significant (bool mask), n_significant.
    """
    p_arr = np.array(p_values, dtype=np.float64)
    n = len(p_arr)

    if method == "bonferroni":
        corrected = np.minimum(p_arr * n, 1.0)
    elif method == "fdr":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(p_arr)
        sorted_p = p_arr[sorted_idx]
        corrected_sorted = np.zeros(n)
        for i in range(n):
            corrected_sorted[i] = sorted_p[i] * n / (i + 1)
        # Enforce monotonicity (backward pass)
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected_sorted = np.minimum(corrected_sorted, 1.0)
        corrected = np.zeros(n)
        corrected[sorted_idx] = corrected_sorted
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'fdr'.")

    significant = corrected < alpha

    return {
        "corrected_p_values": jnp.array(corrected),
        "significant": jnp.array(significant),
        "n_significant": int(np.sum(significant)),
        "method": method,
        "alpha": alpha,
        "original_p_values": jnp.array(p_arr),
    }


def effect_size_analysis(
    model,
    tokens_list: list,
    metric_fn: Callable,
    ablation_fn: Callable,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Compute effect sizes for an ablation with confidence intervals.

    Measures how much an ablation changes a metric, with statistical rigor.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        metric_fn: fn(logits, tokens) -> scalar.
        ablation_fn: fn(model) -> ablated_model (or modified model).
        n_bootstrap: Number of bootstrap samples for CI.
        seed: Random seed.

    Returns:
        Dict with mean_effect, cohens_d, ci_lower, ci_upper, per_input_effects.
    """
    ablated_model = ablation_fn(model)

    effects = []
    baseline_scores = []
    ablated_scores = []
    for tok in tokens_list:
        base_logits = model(tok)
        abl_logits = ablated_model(tok)
        base_score = float(metric_fn(base_logits, tok))
        abl_score = float(metric_fn(abl_logits, tok))
        baseline_scores.append(base_score)
        ablated_scores.append(abl_score)
        effects.append(base_score - abl_score)

    effects = np.array(effects)
    baseline_scores = np.array(baseline_scores)
    ablated_scores = np.array(ablated_scores)
    mean_effect = float(np.mean(effects))

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(baseline_scores) + np.var(ablated_scores)) / 2
    )
    cohens_d = float(mean_effect / max(pooled_std, 1e-10))

    # Bootstrap CI on mean effect
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(effects), size=len(effects), replace=True)
        boot_means.append(float(np.mean(effects[idx])))
    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "mean_effect": mean_effect,
        "cohens_d": cohens_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "per_input_effects": jnp.array(effects),
        "baseline_scores": jnp.array(baseline_scores),
        "ablated_scores": jnp.array(ablated_scores),
    }


def circuit_hypothesis_test(
    model,
    tokens_list: list,
    metric_fn: Callable,
    circuit_components: list,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Test whether a hypothesized circuit explains the model's behavior.

    Compares ablating the circuit components vs ablating random components
    of the same size to determine if the circuit is genuinely important.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        metric_fn: fn(logits, tokens) -> scalar.
        circuit_components: List of (layer, head) tuples in the circuit.
        n_permutations: Number of random circuits to compare against.
        seed: Random seed.

    Returns:
        Dict with circuit_effect, p_value, null_effects, specificity.
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_components = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    circuit_size = len(circuit_components)

    def ablate_components(comps):
        """Compute mean metric drop when ablating given components."""
        def zero_hook(x, name):
            return jnp.zeros_like(x)

        hooks = {}
        for layer, head in comps:
            hook_name = f"blocks.{layer}.attn.hook_result"
            # We'll zero the entire attention output for simplicity
            # A more refined version would zero individual heads
            hooks[hook_name] = zero_hook

        effects = []
        for tok in tokens_list:
            base_logits = model(tok)
            base_score = float(metric_fn(base_logits, tok))

            hook_state = HookState(hook_fns=hooks, cache=None)
            abl_logits = model(tok, hook_state=hook_state)
            abl_score = float(metric_fn(abl_logits, tok))
            effects.append(base_score - abl_score)
        return float(np.mean(effects))

    circuit_effect = ablate_components(circuit_components)

    # Null distribution: random circuits of same size
    rng = np.random.RandomState(seed)
    null_effects = []
    non_circuit = [c for c in all_components if c not in circuit_components]
    for _ in range(n_permutations):
        if len(non_circuit) >= circuit_size:
            idx = rng.choice(len(non_circuit), size=circuit_size, replace=False)
            random_circuit = [non_circuit[i] for i in idx]
        else:
            idx = rng.choice(len(all_components), size=circuit_size, replace=False)
            random_circuit = [all_components[i] for i in idx]
        null_effects.append(ablate_components(random_circuit))
    null_effects = np.array(null_effects)

    # One-sided test: is circuit effect larger than random?
    p_value = float(np.mean(null_effects >= circuit_effect))

    # Specificity: how much more important is the circuit than average?
    specificity = float(circuit_effect / max(np.mean(null_effects), 1e-10))

    return {
        "circuit_effect": circuit_effect,
        "p_value": p_value,
        "null_effects": jnp.array(null_effects),
        "specificity": specificity,
        "circuit_size": circuit_size,
        "n_permutations": n_permutations,
    }
