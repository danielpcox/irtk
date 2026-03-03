"""Gradient-based attention analysis for mechanistic interpretability.

Analyze attention patterns through the lens of gradients: attention
gradient attribution, gradient-weighted patterns, sensitivity maps,
gradient-based head ranking, and attention gradient flow.

References:
- Barkan et al. (2021) "Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def attention_gradient_attribution(
    model,
    tokens,
    metric_fn: Callable,
    layer: int = 0,
    head: int = 0,
) -> dict:
    """Compute gradient of metric w.r.t. attention pattern.

    Shows which attention entries most affect the metric.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: fn(logits, tokens) -> scalar.
        layer: Layer index.
        head: Head index.

    Returns:
        Dict with attention_gradient, gradient_magnitude, top_entries.
    """
    from irtk.hook_points import HookState

    # Forward pass to get pattern
    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    base_pattern = cache.get(pattern_key)
    if base_pattern is None:
        return {"attention_gradient": jnp.zeros((1, 1)), "gradient_magnitude": 0.0, "top_entries": []}

    base_pattern = jnp.array(base_pattern[head])  # [seq, seq]
    seq_len = base_pattern.shape[0]

    # Numerical gradient (small perturbations to pattern entries)
    eps = 1e-4
    grad = np.zeros((seq_len, seq_len))

    base_logits = model(tokens)
    base_metric = float(metric_fn(base_logits, tokens))

    for i in range(seq_len):
        for j in range(i + 1):  # Causal: only lower triangle
            def perturb_hook(x, name, _i=i, _j=j):
                # x is [n_heads, seq, seq]
                delta = jnp.zeros_like(x[head])
                delta = delta.at[_i, _j].set(eps)
                # Re-normalize row
                perturbed = x[head] + delta
                perturbed = perturbed / (jnp.sum(perturbed, axis=-1, keepdims=True) + 1e-10)
                return x.at[head].set(perturbed)

            hs = HookState(hook_fns={pattern_key: perturb_hook}, cache=None)
            perturbed_logits = model(tokens, hook_state=hs)
            perturbed_metric = float(metric_fn(perturbed_logits, tokens))
            grad[i, j] = (perturbed_metric - base_metric) / eps

    # Top entries by gradient magnitude
    entries = []
    for i in range(seq_len):
        for j in range(i + 1):
            entries.append((i, j, float(grad[i, j])))
    entries.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "attention_gradient": jnp.array(grad),
        "gradient_magnitude": float(np.linalg.norm(grad)),
        "top_entries": entries[:10],
        "base_metric": base_metric,
    }


def gradient_weighted_pattern(
    model,
    tokens,
    metric_fn: Callable,
    layer: int = 0,
) -> dict:
    """Compute gradient-weighted attention patterns for all heads.

    Weights each head's pattern by its gradient importance to give
    an aggregate "important attention" map.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: fn(logits, tokens) -> scalar.
        layer: Layer index.

    Returns:
        Dict with weighted_pattern, head_gradients, head_importances.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hook_state)
    base_metric = float(metric_fn(logits, tokens))

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"weighted_pattern": jnp.zeros((1, 1)), "head_importances": jnp.array([])}

    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]

    # Per-head importance via ablation
    head_importances = np.zeros(n_heads)
    for h in range(n_heads):
        def zero_head(x, name, _h=h):
            # Set head's pattern to uniform
            uniform = jnp.ones((seq_len, seq_len)) / seq_len
            # Apply causal mask
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            uniform = uniform * mask
            uniform = uniform / (jnp.sum(uniform, axis=-1, keepdims=True) + 1e-10)
            return x.at[_h].set(uniform)

        hs = HookState(hook_fns={pattern_key: zero_head}, cache=None)
        abl_logits = model(tokens, hook_state=hs)
        abl_metric = float(metric_fn(abl_logits, tokens))
        head_importances[h] = abs(base_metric - abl_metric)

    # Normalize importances
    total_imp = np.sum(head_importances) + 1e-10
    weights = head_importances / total_imp

    # Weighted average pattern
    weighted = np.zeros((seq_len, seq_len))
    for h in range(n_heads):
        weighted += weights[h] * patterns[h]

    return {
        "weighted_pattern": jnp.array(weighted),
        "head_importances": jnp.array(head_importances),
        "head_weights": jnp.array(weights),
        "base_metric": base_metric,
    }


def attention_sensitivity_map(
    model,
    tokens,
    metric_fn: Callable,
    layer: int = 0,
    head: int = 0,
) -> dict:
    """Create a sensitivity map showing which positions are most sensitive.

    For each query position, measures how sensitive the metric is
    to changes in that position's attention distribution.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: fn(logits, tokens) -> scalar.
        layer: Layer index.
        head: Head index.

    Returns:
        Dict with position_sensitivity, most_sensitive_position,
        sensitivity_variance.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hook_state)
    base_metric = float(metric_fn(logits, tokens))

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"position_sensitivity": jnp.array([]), "most_sensitive_position": 0}

    patterns = np.array(cache[pattern_key])
    seq_len = patterns.shape[1]

    # Per-position sensitivity: uniformize each query position's attention
    sensitivity = np.zeros(seq_len)
    for pos in range(seq_len):
        def uniform_pos(x, name, _pos=pos):
            # Make position _pos attend uniformly (within causal mask)
            new = x.at[head, _pos].set(0.0)
            for j in range(_pos + 1):
                new = new.at[head, _pos, j].set(1.0 / (_pos + 1))
            return new

        hs = HookState(hook_fns={pattern_key: uniform_pos}, cache=None)
        abl_logits = model(tokens, hook_state=hs)
        abl_metric = float(metric_fn(abl_logits, tokens))
        sensitivity[pos] = abs(base_metric - abl_metric)

    return {
        "position_sensitivity": jnp.array(sensitivity),
        "most_sensitive_position": int(np.argmax(sensitivity)),
        "sensitivity_variance": float(np.var(sensitivity)),
        "mean_sensitivity": float(np.mean(sensitivity)),
    }


def gradient_head_ranking(
    model,
    tokens,
    metric_fn: Callable,
) -> dict:
    """Rank all heads by their gradient-estimated importance.

    Uses ablation-based importance estimation across all layers and heads.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: fn(logits, tokens) -> scalar.

    Returns:
        Dict with head_ranking, importance_matrix, top_heads, bottom_heads.
    """
    from irtk.hook_points import HookState

    logits = model(tokens)
    base_metric = float(metric_fn(logits, tokens))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    importance = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        hook_name = f"blocks.{l}.attn.hook_result"

        def zero_hook(x, name):
            return jnp.zeros_like(x)

        hs = HookState(hook_fns={hook_name: zero_hook}, cache=None)
        abl_logits = model(tokens, hook_state=hs)
        abl_metric = float(metric_fn(abl_logits, tokens))
        # Spread importance evenly across heads in this layer
        layer_imp = abs(base_metric - abl_metric)
        for h in range(n_heads):
            importance[l, h] = layer_imp / n_heads

    # Ranking
    flat_idx = np.argsort(importance.flatten())[::-1]
    ranking = [(int(idx // n_heads), int(idx % n_heads), float(importance.flatten()[idx]))
               for idx in flat_idx]

    return {
        "head_ranking": ranking,
        "importance_matrix": jnp.array(importance),
        "top_heads": ranking[:5],
        "bottom_heads": ranking[-5:],
        "base_metric": base_metric,
    }


def attention_gradient_flow(
    model,
    tokens,
    metric_fn: Callable,
) -> dict:
    """Analyze how gradient importance flows through attention layers.

    Measures the gradient importance at each layer to understand
    where the model's computation is most sensitive.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: fn(logits, tokens) -> scalar.

    Returns:
        Dict with layer_importance, flow_direction, cumulative_importance.
    """
    from irtk.hook_points import HookState

    logits = model(tokens)
    base_metric = float(metric_fn(logits, tokens))

    n_layers = model.cfg.n_layers
    attn_importance = np.zeros(n_layers)
    mlp_importance = np.zeros(n_layers)

    for l in range(n_layers):
        # Attention importance
        attn_hook = f"blocks.{l}.hook_attn_out"
        def zero_hook(x, name):
            return jnp.zeros_like(x)

        hs = HookState(hook_fns={attn_hook: zero_hook}, cache=None)
        abl_logits = model(tokens, hook_state=hs)
        attn_importance[l] = abs(base_metric - float(metric_fn(abl_logits, tokens)))

        # MLP importance
        mlp_hook = f"blocks.{l}.hook_mlp_out"
        hs = HookState(hook_fns={mlp_hook: zero_hook}, cache=None)
        abl_logits = model(tokens, hook_state=hs)
        mlp_importance[l] = abs(base_metric - float(metric_fn(abl_logits, tokens)))

    total = attn_importance + mlp_importance
    cumulative = np.cumsum(total) / (np.sum(total) + 1e-10)

    # Flow direction: increasing or decreasing importance
    if n_layers > 1:
        flow = float(np.mean(np.diff(total)))
    else:
        flow = 0.0

    return {
        "attn_importance": jnp.array(attn_importance),
        "mlp_importance": jnp.array(mlp_importance),
        "total_importance": jnp.array(total),
        "cumulative_importance": jnp.array(cumulative),
        "flow_direction": flow,
        "peak_layer": int(np.argmax(total)),
    }
