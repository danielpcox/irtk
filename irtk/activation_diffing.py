"""Activation diffing between runs for mechanistic interpretability.

Compare activations between two inputs or two models: paired comparison,
change localization, divergence mapping, causal responsibility, and
minimal change identification.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def _get_all_caches(model, tokens):
    """Run model and return full cache."""
    from irtk.hook_points import HookState
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)
    return cache


def paired_activation_comparison(
    model,
    tokens_a,
    tokens_b,
    layers: Optional[list] = None,
    pos: int = -1,
) -> dict:
    """Compare activations between two inputs at each layer.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input.
        layers: Layers to compare.
        pos: Position.

    Returns:
        Dict with per_layer diffs (cosine, L2, max_dim_diff).
    """
    cache_a = _get_all_caches(model, tokens_a)
    cache_b = _get_all_caches(model, tokens_b)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        key = f"blocks.{l}.hook_resid_post"
        if key in cache_a and key in cache_b:
            a = np.array(cache_a[key][pos])
            b = np.array(cache_b[key][pos])

            l2_dist = float(np.linalg.norm(a - b))
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            max_dim = int(np.argmax(np.abs(a - b)))
            max_diff = float(np.max(np.abs(a - b)))

            per_layer.append({
                "layer": l,
                "l2_distance": l2_dist,
                "cosine_similarity": cos,
                "max_dim_diff": (max_dim, max_diff),
            })

    return {
        "per_layer": per_layer,
        "n_layers": len(layers),
    }


def change_localization(
    model,
    tokens_a,
    tokens_b,
    pos: int = -1,
) -> dict:
    """Localize where the biggest changes happen between two inputs.

    Identifies which layers and components show the largest activation
    differences.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input.
        pos: Position.

    Returns:
        Dict with component_changes, most_changed_component,
        change_by_type.
    """
    cache_a = _get_all_caches(model, tokens_a)
    cache_b = _get_all_caches(model, tokens_b)

    changes = []
    attn_total = 0.0
    mlp_total = 0.0

    for l in range(model.cfg.n_layers):
        for ctype, suffix in [("attn", "hook_attn_out"), ("mlp", "hook_mlp_out")]:
            key = f"blocks.{l}.{suffix}"
            if key in cache_a and key in cache_b:
                a = np.array(cache_a[key][pos])
                b = np.array(cache_b[key][pos])
                diff = float(np.linalg.norm(a - b))
                changes.append({"component": f"{ctype}_{l}", "diff": diff})
                if ctype == "attn":
                    attn_total += diff
                else:
                    mlp_total += diff

    changes.sort(key=lambda x: -x["diff"])
    most_changed = changes[0]["component"] if changes else "none"

    return {
        "component_changes": changes,
        "most_changed_component": most_changed,
        "attn_total_change": attn_total,
        "mlp_total_change": mlp_total,
    }


def divergence_mapping(
    model,
    tokens_a,
    tokens_b,
) -> dict:
    """Map how activations diverge across all positions and layers.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input.

    Returns:
        Dict with divergence_matrix (layers x positions), peak_divergence,
        divergence_onset_layer.
    """
    cache_a = _get_all_caches(model, tokens_a)
    cache_b = _get_all_caches(model, tokens_b)

    seq_len = min(len(tokens_a), len(tokens_b))
    n_layers = model.cfg.n_layers

    div_matrix = np.zeros((n_layers, seq_len))

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache_a and key in cache_b:
            a = np.array(cache_a[key])[:seq_len]
            b = np.array(cache_b[key])[:seq_len]
            for s in range(seq_len):
                div_matrix[l, s] = float(np.linalg.norm(a[s] - b[s]))

    # Peak divergence
    peak_idx = np.unravel_index(np.argmax(div_matrix), div_matrix.shape)
    peak = (int(peak_idx[0]), int(peak_idx[1]), float(div_matrix[peak_idx]))

    # Onset: first layer where mean divergence exceeds threshold
    mean_per_layer = np.mean(div_matrix, axis=1)
    threshold = float(np.mean(mean_per_layer)) * 0.5
    onset = 0
    for l in range(n_layers):
        if mean_per_layer[l] > threshold:
            onset = l
            break

    return {
        "divergence_matrix": jnp.array(div_matrix),
        "peak_divergence": peak,
        "divergence_onset_layer": onset,
        "mean_per_layer": jnp.array(mean_per_layer),
    }


def causal_change_attribution(
    model,
    tokens_a,
    tokens_b,
    metric_fn: Callable,
) -> dict:
    """Attribute output changes to specific components.

    For each component, measures how much of the output difference
    it accounts for by patching from run A to run B.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input (counterfactual).
        metric_fn: fn(logits, tokens) -> scalar.

    Returns:
        Dict with component_attributions, total_change, top_components.
    """
    from irtk.hook_points import HookState

    cache_b = _get_all_caches(model, tokens_b)

    logits_a = model(tokens_a)
    logits_b = model(tokens_b)
    base_metric = float(metric_fn(logits_a, tokens_a))
    cf_metric = float(metric_fn(logits_b, tokens_b))
    total_change = cf_metric - base_metric

    attributions = []

    for l in range(model.cfg.n_layers):
        for ctype, suffix in [("attn", "hook_attn_out"), ("mlp", "hook_mlp_out")]:
            key = f"blocks.{l}.{suffix}"
            if key in cache_b:
                cf_act = jnp.array(cache_b[key])

                def patch_hook(x, name, _cf=cf_act):
                    return _cf

                hook_state = HookState(hook_fns={key: patch_hook}, cache=None)
                patched_logits = model(tokens_a, hook_state=hook_state)
                patched_metric = float(metric_fn(patched_logits, tokens_a))
                attribution = patched_metric - base_metric
                attributions.append({
                    "component": f"{ctype}_{l}",
                    "attribution": attribution,
                    "fraction": attribution / max(abs(total_change), 1e-10),
                })

    attributions.sort(key=lambda x: -abs(x["attribution"]))

    return {
        "component_attributions": attributions,
        "total_change": total_change,
        "top_components": attributions[:5],
    }


def minimal_change_identification(
    model,
    tokens_a,
    tokens_b,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Identify the minimal set of dimensions that explain the output change.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input.
        pos: Position.
        top_k: Number of top dimensions.

    Returns:
        Dict with top_dimensions, dimension_diffs, cumulative_explanation.
    """
    cache_a = _get_all_caches(model, tokens_a)
    cache_b = _get_all_caches(model, tokens_b)

    # Compare final layer residuals
    last_layer = model.cfg.n_layers - 1
    key = f"blocks.{last_layer}.hook_resid_post"

    a = np.array(cache_a[key][pos])
    b = np.array(cache_b[key][pos])
    diff = b - a
    total_diff_norm = float(np.linalg.norm(diff))

    # Rank dimensions by contribution
    dim_diffs = np.abs(diff)
    top_dims = np.argsort(dim_diffs)[::-1][:top_k]

    # Cumulative explanation
    cumulative = []
    cum_norm_sq = 0
    total_sq = float(np.sum(diff ** 2))
    for d in top_dims:
        cum_norm_sq += diff[d] ** 2
        cumulative.append(float(cum_norm_sq / max(total_sq, 1e-10)))

    return {
        "top_dimensions": [(int(d), float(diff[d])) for d in top_dims],
        "dimension_diffs": jnp.array(diff),
        "cumulative_explanation": cumulative,
        "total_diff_norm": total_diff_norm,
    }
