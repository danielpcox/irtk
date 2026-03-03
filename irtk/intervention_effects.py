"""Intervention effect analysis.

Measures how models respond to various types of interventions: activation
scaling, direction addition, component knockouts, and their recovery patterns.

References:
    Turner et al. (2023) "Activation Addition: Steering Language Models Without Optimization"
    Li et al. (2023) "Inference-Time Intervention"
"""

import jax
import jax.numpy as jnp
import numpy as np


def activation_scaling_sensitivity(model, tokens, metric_fn, layer=0, scales=None):
    """Measure metric sensitivity to scaling residual stream activations.

    Scales the residual stream at a given layer by different factors and
    measures the effect on the metric.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        layer: Layer to scale at.
        scales: Array of scale factors (default: [0.0, 0.5, 1.0, 1.5, 2.0]).

    Returns:
        dict with:
            scales: array of scale factors used
            metrics: array of metric values at each scale
            baseline_metric: float (scale=1.0)
            sensitivity: float, |d_metric/d_scale| at scale=1.0 (finite diff)
            monotonic: bool, whether metric changes monotonically with scale
    """
    from irtk.hook_points import HookState

    if scales is None:
        scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    scales = np.array(scales)

    hook_name = f"blocks.{layer}.hook_resid_pre"
    metrics = np.zeros(len(scales))

    for i, s in enumerate(scales):
        def make_scale_fn(scale):
            def fn(x, name):
                return x * scale
            return fn

        state = HookState(hook_fns={hook_name: make_scale_fn(float(s))}, cache={})
        logits = model(tokens, hook_state=state)
        metrics[i] = metric_fn(logits)

    baseline_idx = np.argmin(np.abs(scales - 1.0))
    baseline = metrics[baseline_idx]

    # Sensitivity via finite diff around scale=1.0
    if len(scales) >= 2:
        # Use two neighboring points
        if baseline_idx > 0 and baseline_idx < len(scales) - 1:
            ds = scales[baseline_idx + 1] - scales[baseline_idx - 1]
            dm = metrics[baseline_idx + 1] - metrics[baseline_idx - 1]
            sensitivity = abs(dm / ds) if abs(ds) > 1e-10 else 0.0
        else:
            sensitivity = abs(metrics[-1] - metrics[0]) / (scales[-1] - scales[0] + 1e-10)
    else:
        sensitivity = 0.0

    # Check monotonicity
    diffs = np.diff(metrics)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    return {
        "scales": scales,
        "metrics": metrics,
        "baseline_metric": float(baseline),
        "sensitivity": float(sensitivity),
        "monotonic": monotonic,
    }


def direction_addition_sweep(model, tokens, direction, metric_fn, layer=0, pos=-1,
                              coefficients=None):
    """Sweep the coefficient of a direction added to the residual stream.

    Adds alpha * direction to the residual stream and measures the effect.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: Vector [d_model] to add.
        metric_fn: Function from logits -> scalar.
        layer: Layer to add at.
        pos: Position to add at.
        coefficients: Array of alpha values.

    Returns:
        dict with:
            coefficients: array of alpha values
            metrics: array of metric at each alpha
            baseline_metric: float (alpha=0)
            optimal_coefficient: float, alpha maximizing metric
            effect_range: float, max - min metric across coefficients
    """
    from irtk.hook_points import HookState

    if coefficients is None:
        coefficients = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    coefficients = np.array(coefficients)
    direction = jnp.array(direction)

    hook_name = f"blocks.{layer}.hook_resid_pre"
    metrics = np.zeros(len(coefficients))

    for i, alpha in enumerate(coefficients):
        def make_add_fn(a, d, p):
            def fn(x, name):
                return x.at[p].add(a * d)
            return fn

        state = HookState(
            hook_fns={hook_name: make_add_fn(float(alpha), direction, pos)},
            cache={},
        )
        logits = model(tokens, hook_state=state)
        metrics[i] = metric_fn(logits)

    zero_idx = np.argmin(np.abs(coefficients))
    baseline = metrics[zero_idx]
    optimal_idx = np.argmax(metrics)

    return {
        "coefficients": coefficients,
        "metrics": metrics,
        "baseline_metric": float(baseline),
        "optimal_coefficient": float(coefficients[optimal_idx]),
        "effect_range": float(np.max(metrics) - np.min(metrics)),
    }


def component_knockout_recovery(model, tokens, metric_fn):
    """Measure how well the model recovers from knocking out each component.

    For each component, compares: (a) metric with component knocked out, and
    (b) metric with only that component kept (all others knocked out).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            knockout_effects: array [n_layers, 2] for [attn_effect, mlp_effect] per layer
            solo_metrics: array [n_layers, 2] for [attn_solo, mlp_solo] per layer
            baseline_metric: float
            most_essential: tuple identifying component with largest knockout effect
            most_self_sufficient: tuple, component with best solo performance
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    baseline_logits = model(tokens)
    baseline = metric_fn(baseline_logits)

    knockout_effects = np.zeros((n_layers, 2))  # [attn, mlp]
    solo_metrics = np.zeros((n_layers, 2))

    for layer in range(n_layers):
        # Knock out attention
        attn_key = f"blocks.{layer}.hook_attn_out"
        def zero_fn(x, name):
            return jnp.zeros_like(x)

        state = HookState(hook_fns={attn_key: zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        knockout_effects[layer, 0] = baseline - metric_fn(logits)

        # Knock out MLP
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        state = HookState(hook_fns={mlp_key: zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        knockout_effects[layer, 1] = baseline - metric_fn(logits)

        # Solo: zero everything else at this layer
        # Approximate by zeroing both attn and mlp at all OTHER layers
        other_hooks = {}
        for other_l in range(n_layers):
            if other_l != layer:
                other_hooks[f"blocks.{other_l}.hook_attn_out"] = zero_fn
                other_hooks[f"blocks.{other_l}.hook_mlp_out"] = zero_fn

        # Attn solo: keep attn, zero mlp at this layer too
        attn_solo_hooks = dict(other_hooks)
        attn_solo_hooks[mlp_key] = zero_fn
        state = HookState(hook_fns=attn_solo_hooks, cache={})
        logits = model(tokens, hook_state=state)
        solo_metrics[layer, 0] = metric_fn(logits)

        # MLP solo
        mlp_solo_hooks = dict(other_hooks)
        mlp_solo_hooks[attn_key] = zero_fn
        state = HookState(hook_fns=mlp_solo_hooks, cache={})
        logits = model(tokens, hook_state=state)
        solo_metrics[layer, 1] = metric_fn(logits)

    # Most essential
    max_ko = np.max(np.abs(knockout_effects))
    ko_idx = np.unravel_index(np.argmax(np.abs(knockout_effects)), knockout_effects.shape)
    most_essential = ("attn" if ko_idx[1] == 0 else "mlp", int(ko_idx[0]))

    max_solo = np.max(solo_metrics)
    solo_idx = np.unravel_index(np.argmax(solo_metrics), solo_metrics.shape)
    most_sufficient = ("attn" if solo_idx[1] == 0 else "mlp", int(solo_idx[0]))

    return {
        "knockout_effects": knockout_effects,
        "solo_metrics": solo_metrics,
        "baseline_metric": float(baseline),
        "most_essential": most_essential,
        "most_self_sufficient": most_sufficient,
    }


def intervention_transferability(model, tokens_a, tokens_b, metric_fn, layer=0):
    """Test whether an intervention effective on input A transfers to input B.

    Computes the activation difference between A and B at a layer,
    then patches this difference onto B to see if it shifts B's metric
    toward A's metric.

    Args:
        model: HookedTransformer model.
        tokens_a: First input [seq_len].
        tokens_b: Second input [seq_len].
        metric_fn: Function from logits -> scalar.
        layer: Layer to intervene at.

    Returns:
        dict with:
            metric_a: float
            metric_b: float
            metric_b_patched: float (B with A's activations patched in)
            transfer_fraction: float, how much of the gap is closed
            activation_distance: float, norm of the difference vector
    """
    from irtk.hook_points import HookState

    # Get metrics and activations
    state_a = HookState(hook_fns={}, cache={})
    logits_a = model(tokens_a, hook_state=state_a)
    metric_a = metric_fn(logits_a)

    state_b = HookState(hook_fns={}, cache={})
    logits_b = model(tokens_b, hook_state=state_b)
    metric_b = metric_fn(logits_b)

    hook_name = f"blocks.{layer}.hook_resid_pre"
    resid_a = state_a.cache.get(hook_name)
    resid_b = state_b.cache.get(hook_name)

    if resid_a is not None and resid_b is not None:
        diff = resid_a - resid_b
        distance = float(jnp.linalg.norm(diff))

        # Patch: replace B's residual with A's
        def patch_fn(x, name):
            return resid_a
        state = HookState(hook_fns={hook_name: patch_fn}, cache={})
        patched_logits = model(tokens_b, hook_state=state)
        metric_patched = metric_fn(patched_logits)
    else:
        distance = 0.0
        metric_patched = metric_b

    gap = metric_a - metric_b
    transfer = (metric_patched - metric_b) / (gap + 1e-10) if abs(gap) > 1e-10 else 0.0

    return {
        "metric_a": float(metric_a),
        "metric_b": float(metric_b),
        "metric_b_patched": float(metric_patched),
        "transfer_fraction": float(transfer),
        "activation_distance": float(distance),
    }


def multi_layer_knockout(model, tokens, metric_fn, max_layers=None):
    """Test robustness by knocking out increasing numbers of layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        max_layers: Max layers to knock out (None = all).

    Returns:
        dict with:
            n_knocked_out: array of how many layers knocked out
            metrics: array of metric at each level of knockout
            baseline_metric: float
            half_performance_threshold: int, n layers before metric drops to 50%
            graceful_degradation: bool, True if metric degrades smoothly
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if max_layers is None:
        max_layers = n_layers

    baseline_logits = model(tokens)
    baseline = metric_fn(baseline_logits)

    # Get importance ranking
    effects = []
    for layer in range(n_layers):
        def zero_fn(x, name):
            return jnp.zeros_like(x)
        hooks = {
            f"blocks.{layer}.hook_attn_out": zero_fn,
            f"blocks.{layer}.hook_mlp_out": zero_fn,
        }
        state = HookState(hook_fns=hooks, cache={})
        logits = model(tokens, hook_state=state)
        effect = abs(baseline - metric_fn(logits))
        effects.append((layer, effect))

    # Sort by least important first (knock out least important first)
    effects.sort(key=lambda x: x[1])

    n_levels = min(max_layers, n_layers) + 1
    n_ko = np.arange(n_levels)
    metrics = np.zeros(n_levels)
    metrics[0] = baseline

    for i in range(1, n_levels):
        def zero_fn(x, name):
            return jnp.zeros_like(x)
        hooks = {}
        for j in range(i):
            l = effects[j][0]
            hooks[f"blocks.{l}.hook_attn_out"] = zero_fn
            hooks[f"blocks.{l}.hook_mlp_out"] = zero_fn
        state = HookState(hook_fns=hooks, cache={})
        logits = model(tokens, hook_state=state)
        metrics[i] = metric_fn(logits)

    # Half performance threshold
    half_threshold = n_levels - 1
    if abs(baseline) > 1e-10:
        for i in range(n_levels):
            if abs(metrics[i]) < 0.5 * abs(baseline):
                half_threshold = i
                break

    # Graceful degradation: check if changes are roughly monotonic
    diffs = np.diff(np.abs(metrics))
    graceful = bool(np.all(diffs <= 0.1 * abs(baseline) + 1e-5))

    return {
        "n_knocked_out": n_ko,
        "metrics": metrics,
        "baseline_metric": float(baseline),
        "half_performance_threshold": int(half_threshold),
        "graceful_degradation": graceful,
    }
