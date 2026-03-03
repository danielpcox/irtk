"""Robustness and perturbation analysis.

Analyzes model robustness by injecting noise into weights and activations.
Identifies critical parameters, measures noise propagation, and classifies
circuits as brittle vs robust.

Functions:
- weight_noise_tolerance: Measure performance degradation under weight noise
- critical_parameter_identification: Find parameters fragile to perturbation
- activation_noise_propagation: Track how noise amplifies through layers
- mode_connectivity_probe: Test loss landscape smoothness between weight configs
- brittle_vs_robust_circuits: Classify circuits by noise tolerance

References:
    - Li et al. (2018) "Measuring the Intrinsic Dimension of Objective Landscapes"
    - Frankle & Carlin (2019) "Lottery Ticket Hypothesis"
    - Burns et al. (2023) "Weak-to-Strong Generalization"
"""

from typing import Optional, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def weight_noise_tolerance(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    noise_scales: Optional[list] = None,
    seed: int = 0,
) -> dict:
    """Measure performance degradation under weight noise.

    Adds Gaussian noise to all weights at various scales and measures
    the resulting metric change.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        noise_scales: List of noise standard deviations. Defaults to geometric range.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
            "noise_scales": list of noise levels tested
            "metric_values": [n_scales] metric at each noise level
            "metric_drops": [n_scales] absolute metric change from clean
            "tolerance_threshold": largest noise scale with <10% metric change
            "clean_metric": metric on unperturbed model
    """
    if noise_scales is None:
        noise_scales = [0.001, 0.005, 0.01, 0.05, 0.1]

    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    metrics = []
    key = jax.random.PRNGKey(seed)

    for scale in noise_scales:
        # Add noise to all parameters
        leaves, treedef = jax.tree.flatten(model)
        noisy_leaves = []
        for leaf in leaves:
            if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * scale
                noisy_leaves.append(leaf + noise)
            else:
                noisy_leaves.append(leaf)
        noisy_model = jax.tree.unflatten(treedef, noisy_leaves)
        noisy_logits = noisy_model(tokens)
        metrics.append(float(metric_fn(noisy_logits)))

    metrics = np.array(metrics)
    drops = np.abs(metrics - clean_metric)

    # Tolerance: largest scale with <10% change
    threshold_pct = 0.1 * abs(clean_metric) if abs(clean_metric) > 1e-10 else 0.1
    tolerance = 0.0
    for i, drop in enumerate(drops):
        if drop < threshold_pct:
            tolerance = noise_scales[i]

    return {
        "noise_scales": noise_scales,
        "metric_values": metrics,
        "metric_drops": drops,
        "tolerance_threshold": tolerance,
        "clean_metric": clean_metric,
    }


def critical_parameter_identification(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    noise_scale: float = 0.01,
    seed: int = 0,
) -> dict:
    """Find parameters that are fragile to perturbation.

    Perturbs each weight matrix independently and measures the metric
    change to identify critical vs robust parameters.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        noise_scale: Standard deviation of noise.
        seed: Random seed.

    Returns:
        Dict with:
            "parameter_sensitivity": dict mapping param_name -> sensitivity_score
            "most_critical": name of most sensitive parameter
            "least_critical": name of least sensitive parameter
            "sensitivity_ranking": list of (param_name, score) sorted by sensitivity
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))
    key = jax.random.PRNGKey(seed)

    sensitivities = {}

    # Test each component's weights
    for l in range(model.cfg.n_layers):
        block = model.blocks[l]

        # Attention weights
        for wname in ['W_Q', 'W_K', 'W_V', 'W_O']:
            param_name = f"blocks.{l}.attn.{wname}"
            w = getattr(block.attn, wname, None)
            if w is not None and isinstance(w, jnp.ndarray):
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, w.shape, dtype=w.dtype) * noise_scale

                import equinox as eqx
                noisy_model = eqx.tree_at(
                    lambda m, _l=l, _w=wname: getattr(m.blocks[_l].attn, _w),
                    model, w + noise
                )
                noisy_logits = noisy_model(tokens)
                sensitivities[param_name] = abs(float(metric_fn(noisy_logits)) - clean_metric)

        # MLP weights
        for wname in ['W_in', 'W_out']:
            param_name = f"blocks.{l}.mlp.{wname}"
            w = getattr(block.mlp, wname, None)
            if w is not None and isinstance(w, jnp.ndarray):
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, w.shape, dtype=w.dtype) * noise_scale

                import equinox as eqx
                noisy_model = eqx.tree_at(
                    lambda m, _l=l, _w=wname: getattr(m.blocks[_l].mlp, _w),
                    model, w + noise
                )
                noisy_logits = noisy_model(tokens)
                sensitivities[param_name] = abs(float(metric_fn(noisy_logits)) - clean_metric)

    ranking = sorted(sensitivities.items(), key=lambda x: -x[1])
    most = ranking[0][0] if ranking else ""
    least = ranking[-1][0] if ranking else ""

    return {
        "parameter_sensitivity": sensitivities,
        "most_critical": most,
        "least_critical": least,
        "sensitivity_ranking": ranking,
    }


def activation_noise_propagation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    noise_scale: float = 0.1,
    seed: int = 0,
) -> dict:
    """Track how noise amplifies or dampens through layers.

    Injects noise at each layer's residual stream and measures the output
    perturbation, revealing which layers amplify vs attenuate noise.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        noise_scale: Standard deviation of injected noise.
        seed: Random seed.

    Returns:
        Dict with:
            "injection_layers": [n_layers] layer indices
            "output_perturbations": [n_layers] output perturbation when noise injected at each layer
            "amplification_factors": [n_layers] ratio of output perturbation to input noise
            "most_amplifying_layer": layer that amplifies noise most
            "noise_stability": overall model noise stability score (lower = more stable)
    """
    n_layers = model.cfg.n_layers
    key = jax.random.PRNGKey(seed)

    clean_logits = np.array(model(tokens))

    perturbations = np.zeros(n_layers)

    for l in range(n_layers):
        hook_name = f"blocks.{l}.hook_resid_pre"
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (len(tokens), model.cfg.d_model)) * noise_scale

        def noise_hook(x, name, _noise=noise):
            return x + _noise

        noisy_logits = np.array(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, noise_hook)]))
        perturbations[l] = float(np.linalg.norm(noisy_logits - clean_logits))

    # Amplification: how much bigger is output perturbation vs input noise
    input_noise_norm = float(np.sqrt(len(tokens) * model.cfg.d_model)) * noise_scale
    amplification = perturbations / (input_noise_norm + 1e-10)

    most_amplifying = int(np.argmax(amplification))
    stability = float(np.mean(amplification))

    return {
        "injection_layers": list(range(n_layers)),
        "output_perturbations": perturbations,
        "amplification_factors": amplification,
        "most_amplifying_layer": most_amplifying,
        "noise_stability": stability,
    }


def mode_connectivity_probe(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    n_interpolations: int = 5,
) -> dict:
    """Test loss landscape smoothness between two weight configurations.

    Linearly interpolates between two models' weights and measures the
    metric at each point, testing for barriers in the loss landscape.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        n_interpolations: Number of interpolation points.

    Returns:
        Dict with:
            "alphas": [n_interpolations] interpolation coefficients (0=model_a, 1=model_b)
            "metrics": [n_interpolations] metric at each interpolation
            "is_connected": whether metric stays within 10% of endpoints
            "max_barrier": maximum metric deviation from linear interpolation
            "smoothness": 1 - normalized barrier height (1 = perfectly smooth)
    """
    alphas = np.linspace(0.0, 1.0, n_interpolations)
    metrics = np.zeros(n_interpolations)

    leaves_a, treedef = jax.tree.flatten(model_a)
    leaves_b, _ = jax.tree.flatten(model_b)

    for i, alpha in enumerate(alphas):
        interp_leaves = []
        for la, lb in zip(leaves_a, leaves_b):
            if isinstance(la, jnp.ndarray) and la.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
                interp_leaves.append(la * (1 - alpha) + lb * alpha)
            else:
                interp_leaves.append(la)
        interp_model = jax.tree.unflatten(treedef, interp_leaves)
        logits = interp_model(tokens)
        metrics[i] = float(metric_fn(logits))

    # Linear interpolation of endpoint metrics
    linear_interp = metrics[0] * (1 - alphas) + metrics[-1] * alphas
    deviations = np.abs(metrics - linear_interp)
    max_barrier = float(np.max(deviations))

    # Connected if all points within 10% of endpoint range
    endpoint_range = max(abs(metrics[0]), abs(metrics[-1]), 1e-10)
    connected = bool(max_barrier < 0.1 * endpoint_range)

    smoothness = 1.0 - min(1.0, max_barrier / (endpoint_range + 1e-10))

    return {
        "alphas": alphas,
        "metrics": metrics,
        "is_connected": connected,
        "max_barrier": max_barrier,
        "smoothness": smoothness,
    }


def brittle_vs_robust_circuits(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> dict:
    """Classify circuits as brittle or robust under noise.

    Injects noise into each head's output and measures whether the metric
    degrades gracefully or catastrophically.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        noise_scale: Noise intensity.
        seed: Random seed.

    Returns:
        Dict with:
            "head_robustness": dict mapping (layer, head) -> robustness score
            "brittle_heads": list of (layer, head) with robustness < 0.5
            "robust_heads": list of (layer, head) with robustness >= 0.5
            "mean_robustness": average robustness across all heads
            "most_brittle": (layer, head) tuple of most brittle head
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))
    key = jax.random.PRNGKey(seed)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    robustness = {}

    for l in range(n_layers):
        for h in range(n_heads):
            hook_name = f"blocks.{l}.attn.hook_z"
            key, subkey = jax.random.split(key)

            def noise_head_hook(x, name, _h=h, _key=subkey):
                noise = jax.random.normal(_key, (x.shape[0], d_head)) * noise_scale
                return x.at[:, _h, :].add(noise)

            noisy_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, noise_head_hook)])
            noisy_metric = float(metric_fn(noisy_logits))

            # Robustness = 1 - normalized metric change
            change = abs(noisy_metric - clean_metric) / (abs(clean_metric) + 1e-10)
            robustness[(l, h)] = max(0.0, 1.0 - change)

    brittle = [(l, h) for (l, h), r in robustness.items() if r < 0.5]
    robust = [(l, h) for (l, h), r in robustness.items() if r >= 0.5]
    mean_r = float(np.mean(list(robustness.values())))
    most_brittle = min(robustness, key=robustness.get) if robustness else (0, 0)

    return {
        "head_robustness": robustness,
        "brittle_heads": brittle,
        "robust_heads": robust,
        "mean_robustness": mean_r,
        "most_brittle": most_brittle,
    }
