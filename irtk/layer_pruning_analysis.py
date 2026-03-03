"""Layer pruning analysis.

Analyzes the effect of pruning layers: skip connections, layer criticality,
optimal pruning order, and performance under progressive layer removal.

References:
    Fan et al. (2020) "Reducing Transformer Depth on Demand with Structured Dropout"
    Men et al. (2024) "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"
"""

import jax
import jax.numpy as jnp
import numpy as np


def layer_skip_analysis(model, tokens, metric_fn):
    """Measure the effect of skipping each layer.

    For each layer, bypasses it (identity skip connection) and measures
    the metric change.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            skip_effects: array [n_layers] of metric change when skipping
            most_critical_layer: int
            least_critical_layer: int
            mean_skip_effect: float
            can_skip: array [n_layers] of bool (effect < 10% of baseline)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline = metric_fn(model(tokens))

    effects = np.zeros(n_layers)
    for layer in range(n_layers):
        # Skip this layer by zeroing both attn and mlp outputs
        hooks = {}
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        def zero_fn(x, name):
            return jnp.zeros_like(x)

        hooks[attn_key] = zero_fn
        hooks[mlp_key] = zero_fn

        state = HookState(hook_fns=hooks, cache={})
        logits = model(tokens, hook_state=state)
        effects[layer] = abs(baseline - metric_fn(logits))

    threshold = abs(baseline) * 0.1 if abs(baseline) > 1e-10 else 0.1
    can_skip = effects < threshold

    return {
        "skip_effects": effects,
        "most_critical_layer": int(np.argmax(effects)),
        "least_critical_layer": int(np.argmin(effects)),
        "mean_skip_effect": float(np.mean(effects)),
        "can_skip": can_skip,
    }


def progressive_layer_pruning(model, tokens, metric_fn):
    """Progressively prune layers in order of least importance.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            pruning_order: list of layer indices in pruning order
            metrics_after_pruning: array [n_layers+1] of metric after each pruning step
            layers_before_50pct_loss: int
            graceful_degradation: bool
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline = metric_fn(model(tokens))

    # Get initial effects
    skip = layer_skip_analysis(model, tokens, metric_fn)
    remaining = list(range(n_layers))
    pruned = []
    metrics = [float(baseline)]

    for _ in range(n_layers):
        # Find least critical remaining layer
        best_layer = remaining[0]
        best_effect = float('inf')

        for layer in remaining:
            hooks = {}
            for pl in pruned + [layer]:
                hooks[f"blocks.{pl}.hook_attn_out"] = lambda x, name: jnp.zeros_like(x)
                hooks[f"blocks.{pl}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)

            state = HookState(hook_fns=hooks, cache={})
            logits = model(tokens, hook_state=state)
            effect = abs(baseline - metric_fn(logits))

            if effect < best_effect:
                best_effect = effect
                best_layer = layer

        pruned.append(best_layer)
        remaining.remove(best_layer)
        metrics.append(float(baseline - best_effect) if abs(baseline) > 1e-10 else 0.0)

    metrics = np.array(metrics)

    # Layers before 50% loss
    half_threshold = abs(baseline) * 0.5
    n_before_50 = n_layers
    for i, m in enumerate(metrics[1:]):
        if abs(baseline - m) > half_threshold:
            n_before_50 = i
            break

    # Graceful degradation: monotonic decrease
    graceful = all(abs(metrics[i]) >= abs(metrics[i + 1]) - 1e-5 for i in range(len(metrics) - 1))

    return {
        "pruning_order": pruned,
        "metrics_after_pruning": metrics,
        "layers_before_50pct_loss": n_before_50,
        "graceful_degradation": bool(graceful),
    }


def layer_criticality_profile(model, tokens, metric_fn):
    """Profile each layer's criticality from multiple perspectives.

    Combines skip effect, solo contribution, and position in the network.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            attn_criticality: array [n_layers]
            mlp_criticality: array [n_layers]
            combined_criticality: array [n_layers]
            critical_layers: list of layer indices above mean
            redundant_layers: list of layer indices below 10% of max
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline = metric_fn(model(tokens))

    attn_crit = np.zeros(n_layers)
    mlp_crit = np.zeros(n_layers)

    for layer in range(n_layers):
        # Attn knockout
        def zero_fn(x, name):
            return jnp.zeros_like(x)

        state = HookState(hook_fns={f"blocks.{layer}.hook_attn_out": zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        attn_crit[layer] = abs(baseline - metric_fn(logits))

        # MLP knockout
        state = HookState(hook_fns={f"blocks.{layer}.hook_mlp_out": zero_fn}, cache={})
        logits = model(tokens, hook_state=state)
        mlp_crit[layer] = abs(baseline - metric_fn(logits))

    combined = attn_crit + mlp_crit
    mean_crit = np.mean(combined)
    max_crit = np.max(combined)

    critical = [int(l) for l in range(n_layers) if combined[l] > mean_crit]
    redundant = [int(l) for l in range(n_layers) if combined[l] < max_crit * 0.1]

    return {
        "attn_criticality": attn_crit,
        "mlp_criticality": mlp_crit,
        "combined_criticality": combined,
        "critical_layers": critical,
        "redundant_layers": redundant,
    }


def layer_similarity_for_pruning(model, tokens, pos=-1):
    """Identify similar layers that could be merged or pruned.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            layer_similarity_matrix: array [n_layers, n_layers]
            most_similar_pair: tuple (layer_a, layer_b)
            most_different_pair: tuple (layer_a, layer_b)
            similarity_to_identity: array [n_layers] (how much each layer acts like identity)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    # Collect layer deltas (what each layer adds)
    deltas = []
    for layer in range(n_layers):
        pre_key = f"blocks.{layer}.hook_resid_pre" if layer == 0 else f"blocks.{layer - 1}.hook_resid_post"
        # Actually, resid_pre for layer l and resid_post for layer l
        # For simplicity, use attn_out + mlp_out as the delta
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        delta = np.zeros(model.cfg.d_model)
        if attn is not None:
            delta += np.array(attn[pos])
        if mlp is not None:
            delta += np.array(mlp[pos])
        deltas.append(delta)

    # Similarity matrix
    sim = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            ni = np.linalg.norm(deltas[i]) + 1e-10
            nj = np.linalg.norm(deltas[j]) + 1e-10
            sim[i, j] = float(np.dot(deltas[i], deltas[j]) / (ni * nj))

    # Most similar/different (off-diagonal)
    mask = np.ones((n_layers, n_layers)) - np.eye(n_layers)
    sim_masked = sim * mask - (1 - mask) * 10
    most_sim = np.unravel_index(np.argmax(sim_masked), sim.shape)
    diff_masked = sim * mask + (1 - mask) * 10
    most_diff = np.unravel_index(np.argmin(diff_masked), sim.shape)

    # Similarity to identity (small norm = acts like skip)
    id_sim = np.zeros(n_layers)
    for l in range(n_layers):
        pre_key = "blocks.0.hook_resid_pre" if l == 0 else f"blocks.{l - 1}.hook_resid_post"
        resid = cache.get(pre_key)
        if resid is not None:
            resid_norm = np.linalg.norm(np.array(resid[pos])) + 1e-10
            delta_norm = np.linalg.norm(deltas[l])
            id_sim[l] = 1.0 - delta_norm / resid_norm

    return {
        "layer_similarity_matrix": sim,
        "most_similar_pair": (int(most_sim[0]), int(most_sim[1])),
        "most_different_pair": (int(most_diff[0]), int(most_diff[1])),
        "similarity_to_identity": id_sim,
    }


def optimal_layer_subset(model, tokens, metric_fn, target_layers=None):
    """Find the best subset of layers to keep for a given budget.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.
        target_layers: Number of layers to keep (default: n_layers // 2).

    Returns:
        dict with:
            kept_layers: list of layer indices to keep
            pruned_layers: list of layer indices to prune
            subset_metric: float
            full_metric: float
            retention_ratio: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    if target_layers is None:
        target_layers = max(1, n_layers // 2)
    target_layers = min(target_layers, n_layers)

    baseline = metric_fn(model(tokens))

    # Greedy: remove least important layer one at a time
    remaining = list(range(n_layers))
    pruned = []

    while len(remaining) > target_layers:
        best_layer = remaining[0]
        best_metric_after = -float('inf')

        for layer in remaining:
            hooks = {}
            for pl in pruned + [layer]:
                hooks[f"blocks.{pl}.hook_attn_out"] = lambda x, name: jnp.zeros_like(x)
                hooks[f"blocks.{pl}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)

            state = HookState(hook_fns=hooks, cache={})
            logits = model(tokens, hook_state=state)
            m = abs(metric_fn(logits))

            if m > best_metric_after:
                best_metric_after = m
                best_layer = layer

        pruned.append(best_layer)
        remaining.remove(best_layer)

    # Final metric
    hooks = {}
    for pl in pruned:
        hooks[f"blocks.{pl}.hook_attn_out"] = lambda x, name: jnp.zeros_like(x)
        hooks[f"blocks.{pl}.hook_mlp_out"] = lambda x, name: jnp.zeros_like(x)

    state = HookState(hook_fns=hooks, cache={})
    logits = model(tokens, hook_state=state)
    subset_metric = metric_fn(logits)

    retention = abs(subset_metric) / (abs(baseline) + 1e-10)

    return {
        "kept_layers": sorted(remaining),
        "pruned_layers": sorted(pruned),
        "subset_metric": float(subset_metric),
        "full_metric": float(baseline),
        "retention_ratio": float(retention),
    }
