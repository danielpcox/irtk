"""Layer-level intervention experiments.

Tools for performing targeted interventions at specific layers:
activation addition, scaling experiments, direction interventions,
cross-layer transfer, and intervention sweeps.
"""

import jax
import jax.numpy as jnp
import numpy as np


def activation_addition(model, tokens, direction, layer, scale=1.0, pos=None, metric_fn=None):
    """Add a direction vector to the residual stream at a specific layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: [d_model] vector to add.
        layer: Layer at which to add.
        scale: Scaling factor for the direction.
        pos: Position(s) at which to add (None for all).
        metric_fn: Optional metric function to measure effect.

    Returns:
        dict with:
            original_logits: [seq_len, d_vocab]
            modified_logits: [seq_len, d_vocab]
            logit_change: [seq_len, d_vocab]
            metric_change: float (if metric_fn provided)
            top_promoted_tokens: list of (token, logit_change)
            top_demoted_tokens: list of (token, logit_change)
    """
    from irtk.hook_points import HookState

    direction_arr = np.array(direction)
    d_norm = np.linalg.norm(direction_arr)
    if d_norm > 1e-10:
        direction_arr = direction_arr / d_norm

    original_logits = np.array(model(tokens))

    hook_key = f"blocks.{layer}.hook_resid_pre"
    def make_hook(d, s, p):
        def hook_fn(x, name):
            add = jnp.array(d * s)
            if p is not None:
                return x.at[p, :].add(add)
            else:
                return x + add[None, :]
        return hook_fn

    state = HookState(hook_fns={hook_key: make_hook(direction_arr, scale, pos)}, cache={})
    modified_logits = np.array(model(tokens, hook_state=state))

    logit_change = modified_logits - original_logits
    last_change = logit_change[-1]

    top_promoted = np.argsort(-last_change)[:5]
    top_demoted = np.argsort(last_change)[:5]

    result = {
        "original_logits": original_logits,
        "modified_logits": modified_logits,
        "logit_change": logit_change,
        "top_promoted_tokens": [(int(t), float(last_change[t])) for t in top_promoted],
        "top_demoted_tokens": [(int(t), float(last_change[t])) for t in top_demoted],
    }

    if metric_fn is not None:
        orig_metric = float(metric_fn(original_logits))
        mod_metric = float(metric_fn(modified_logits))
        result["metric_change"] = mod_metric - orig_metric
    else:
        result["metric_change"] = 0.0

    return result


def scaling_experiment(model, tokens, layer, metric_fn, scales=None):
    """Scale the residual stream at a specific layer and measure effects.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer at which to scale.
        metric_fn: Function mapping logits to scalar.
        scales: List of scaling factors to try (default: 0.0 to 2.0).

    Returns:
        dict with:
            scales: list of float
            metrics: list of float (metric at each scale)
            base_metric: float (at scale=1.0)
            sensitivity: float (how much metric changes per unit scale)
            optimal_scale: float (scale that maximizes metric)
    """
    from irtk.hook_points import HookState

    if scales is None:
        scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    metrics = []
    for s in scales:
        if abs(s - 1.0) < 1e-10:
            metrics.append(base_metric)
            continue

        hook_key = f"blocks.{layer}.hook_resid_pre"
        sc = s
        def make_hook(scale):
            def hook_fn(x, name):
                return x * scale
            return hook_fn
        state = HookState(hook_fns={hook_key: make_hook(sc)}, cache={})
        mod_logits = np.array(model(tokens, hook_state=state))
        metrics.append(float(metric_fn(mod_logits)))

    # Sensitivity: approximate derivative at scale=1
    # Use finite differences
    idx_1 = None
    for i, s in enumerate(scales):
        if abs(s - 1.0) < 1e-10:
            idx_1 = i
            break

    sensitivity = 0.0
    if idx_1 is not None:
        if idx_1 > 0:
            sensitivity = (metrics[idx_1] - metrics[idx_1 - 1]) / (scales[idx_1] - scales[idx_1 - 1])
        elif idx_1 < len(scales) - 1:
            sensitivity = (metrics[idx_1 + 1] - metrics[idx_1]) / (scales[idx_1 + 1] - scales[idx_1])

    optimal_idx = int(np.argmax(metrics))
    optimal_scale = scales[optimal_idx]

    return {
        "scales": scales,
        "metrics": metrics,
        "base_metric": base_metric,
        "sensitivity": float(sensitivity),
        "optimal_scale": optimal_scale,
    }


def direction_intervention(model, tokens, direction, metric_fn, layers=None, scales=None):
    """Intervene with a direction at multiple layers and measure effects.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        direction: [d_model] direction vector.
        metric_fn: Function mapping logits to scalar.
        layers: List of layers to try (default: all).
        scales: List of scales to try at each layer.

    Returns:
        dict with:
            layer_effects: [n_layers_tested, n_scales] metric values
            best_layer: int
            best_scale: float
            best_metric: float
            effect_profile: [n_layers_tested] max effect at each layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if layers is None:
        layers = list(range(n_layers))
    if scales is None:
        scales = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    direction_arr = np.array(direction)
    d_norm = np.linalg.norm(direction_arr)
    if d_norm > 1e-10:
        direction_arr = direction_arr / d_norm

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    layer_effects = np.zeros((len(layers), len(scales)))
    best_layer = layers[0]
    best_scale = 0.0
    best_metric = base_metric

    for li, layer in enumerate(layers):
        for si, scale in enumerate(scales):
            if abs(scale) < 1e-10:
                layer_effects[li, si] = base_metric
                continue

            hook_key = f"blocks.{layer}.hook_resid_pre"
            d, s = direction_arr, scale
            def make_hook(d_vec, sc):
                def hook_fn(x, name):
                    return x + jnp.array(d_vec * sc)[None, :]
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(d, s)}, cache={})
            mod_logits = np.array(model(tokens, hook_state=state))
            metric = float(metric_fn(mod_logits))
            layer_effects[li, si] = metric

            if metric > best_metric:
                best_metric = metric
                best_layer = layer
                best_scale = scale

    effect_profile = np.max(np.abs(layer_effects - base_metric), axis=1)

    return {
        "layer_effects": layer_effects,
        "best_layer": best_layer,
        "best_scale": best_scale,
        "best_metric": best_metric,
        "effect_profile": effect_profile,
    }


def cross_layer_transfer(model, tokens, source_layer, target_layer, metric_fn, pos=-1):
    """Transfer activations from one layer to another and measure effects.

    Takes the residual stream state at source_layer and patches it into
    target_layer, testing whether the representation is compatible.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        source_layer: Layer to read activations from.
        target_layer: Layer to write activations to.
        metric_fn: Function mapping logits to scalar.
        pos: Position to transfer (-1 for last).

    Returns:
        dict with:
            base_metric: float
            transferred_metric: float
            metric_change: float
            source_norm: float (norm of source activation)
            target_norm: float (norm of target activation)
            cosine_similarity: float (similarity between source and target)
    """
    from irtk.hook_points import HookState

    # Get base metric and source activations
    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    source_key = f"blocks.{source_layer}.hook_resid_post"
    target_key = f"blocks.{target_layer}.hook_resid_pre"

    source_act = cache.get(source_key)
    target_act = cache.get(target_key)

    if source_act is None or target_act is None:
        return {
            "base_metric": base_metric,
            "transferred_metric": base_metric,
            "metric_change": 0.0,
            "source_norm": 0.0,
            "target_norm": 0.0,
            "cosine_similarity": 0.0,
        }

    s_arr = np.array(source_act)
    t_arr = np.array(target_act)

    source_norm = float(np.linalg.norm(s_arr[pos]))
    target_norm = float(np.linalg.norm(t_arr[pos]))

    if source_norm > 1e-10 and target_norm > 1e-10:
        cosine = float(np.dot(s_arr[pos], t_arr[pos]) / (source_norm * target_norm))
    else:
        cosine = 0.0

    # Transfer: replace target_layer's input with source_layer's output
    src = jnp.array(s_arr)
    p = pos
    def make_hook(source, position):
        def hook_fn(x, name):
            return x.at[position, :].set(source[position, :])
        return hook_fn
    state = HookState(hook_fns={target_key: make_hook(src, p)}, cache={})
    transferred_logits = np.array(model(tokens, hook_state=state))
    transferred_metric = float(metric_fn(transferred_logits))

    return {
        "base_metric": base_metric,
        "transferred_metric": transferred_metric,
        "metric_change": transferred_metric - base_metric,
        "source_norm": source_norm,
        "target_norm": target_norm,
        "cosine_similarity": cosine,
    }


def intervention_sweep(model, tokens, metric_fn, n_directions=5):
    """Sweep random direction interventions across all layers.

    Tests random directions at each layer to build an overall picture
    of intervention sensitivity across the model.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        n_directions: Number of random directions to test per layer.

    Returns:
        dict with:
            layer_sensitivity: [n_layers] mean absolute metric change per layer
            layer_max_effect: [n_layers] maximum effect achievable per layer
            most_sensitive_layer: int
            least_sensitive_layer: int
            total_sensitivity: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    rng = np.random.RandomState(42)
    layer_effects = np.zeros((n_layers, n_directions))

    for layer in range(n_layers):
        for di in range(n_directions):
            direction = rng.randn(d_model).astype(np.float32)
            direction /= np.linalg.norm(direction)

            hook_key = f"blocks.{layer}.hook_resid_pre"
            d = direction
            def make_hook(d_vec):
                def hook_fn(x, name):
                    return x + jnp.array(d_vec)[None, :]
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(d)}, cache={})
            mod_logits = np.array(model(tokens, hook_state=state))
            layer_effects[layer, di] = abs(float(metric_fn(mod_logits)) - base_metric)

    layer_sensitivity = np.mean(layer_effects, axis=1)
    layer_max = np.max(layer_effects, axis=1)

    return {
        "layer_sensitivity": layer_sensitivity,
        "layer_max_effect": layer_max,
        "most_sensitive_layer": int(np.argmax(layer_sensitivity)),
        "least_sensitive_layer": int(np.argmin(layer_sensitivity)),
        "total_sensitivity": float(np.sum(layer_sensitivity)),
    }
