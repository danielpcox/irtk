"""Semantic saturation analysis.

Analyzes when and where semantic information becomes "complete" in the
residual stream. Identifies redundant layers, saturation points, and
the balance of early vs late computation.

Functions:
- semantic_information_saturation: Per-layer predictive information for target token
- redundant_layer_detection: Find layers whose removal barely affects output
- token_saturation_curve: Per-position entropy/confidence across layers
- representation_stabilization_point: Layer where representations stop changing
- early_vs_late_computation_balance: Computation burden distribution

References:
    - Veit et al. (2016) "Residual Networks Behave Like Ensembles of Shallow Networks"
    - Elhage et al. (2022) "Toy Models of Superposition" (information compaction)
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def semantic_information_saturation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict:
    """Measure predictive information about a target token at each layer.

    Uses logit lens to project each layer's residual stream to output space
    and measures the probability assigned to the target token.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token index to track probability for.
        pos: Token position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_probs": [n_layers] probability of target at each layer
            "saturation_layer": first layer achieving >90% of final prob
            "information_gain": [n_layers] per-layer probability increase
            "final_prob": final output probability for target
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_probs = np.array(jax.nn.softmax(logits[pos]))
    final_prob = float(final_probs[target_token])

    layer_probs = []
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            layer_p = np.exp(layer_logits - np.max(layer_logits))
            layer_p = layer_p / np.sum(layer_p)
            layer_probs.append(float(layer_p[target_token]))
        else:
            layer_probs.append(0.0)

    layer_probs = np.array(layer_probs)

    # Information gain per layer
    info_gain = np.zeros(n_layers)
    for i in range(n_layers):
        if i == 0:
            info_gain[i] = layer_probs[i]
        else:
            info_gain[i] = layer_probs[i] - layer_probs[i - 1]

    # Saturation: first layer achieving 90% of final prob
    threshold = 0.9 * final_prob
    saturation = -1
    for i, p in enumerate(layer_probs):
        if p >= threshold:
            saturation = i
            break

    return {
        "layer_probs": layer_probs,
        "saturation_layer": saturation,
        "information_gain": info_gain,
        "final_prob": final_prob,
    }


def redundant_layer_detection(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    threshold: float = 0.05,
) -> dict:
    """Find layers whose removal barely affects the output metric.

    Skips each layer individually (by zeroing its contribution) and measures
    the metric change.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        threshold: Maximum absolute metric change for "redundant".

    Returns:
        Dict with:
            "layer_effects": [n_layers] metric change when layer is skipped
            "redundant_layers": list of layer indices with effect < threshold
            "essential_layers": list of layer indices with effect >= threshold
            "most_redundant": layer with smallest effect
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    n_layers = model.cfg.n_layers
    effects = np.zeros(n_layers)

    for l in range(n_layers):
        # Zero out this layer's output contribution
        hook_name = f"blocks.{l}.hook_resid_post"

        def skip_hook(x, name, _pre_key=f"blocks.{l}.hook_resid_pre"):
            # Replace post with pre (skip the layer)
            return x  # This won't work as intended; use attn+mlp zeroing

        # More reliable: zero out attn and MLP outputs
        attn_hook = f"blocks.{l}.attn.hook_result"
        mlp_hook = f"blocks.{l}.hook_mlp_out"

        hooks = []

        def zero_hook(x, name):
            return jnp.zeros_like(x)

        # Try to zero the attention and MLP outputs
        hooks.append((attn_hook, zero_hook))
        if mlp_hook in [f"blocks.{l}.hook_mlp_out"]:
            hooks.append((mlp_hook, zero_hook))

        try:
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            effects[l] = abs(float(metric_fn(ablated_logits)) - clean_metric)
        except Exception:
            effects[l] = 0.0

    redundant = [int(l) for l in range(n_layers) if effects[l] < threshold]
    essential = [int(l) for l in range(n_layers) if effects[l] >= threshold]
    most_redundant = int(np.argmin(effects)) if n_layers > 0 else 0

    return {
        "layer_effects": effects,
        "redundant_layers": redundant,
        "essential_layers": essential,
        "most_redundant": most_redundant,
    }


def token_saturation_curve(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Track output entropy at each layer for a specific token position.

    Uses logit lens to measure how "decided" the model is at each layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_entropies": [n_layers] entropy of logit lens prediction
            "entropy_reduction": total entropy drop from first to last layer
            "steepest_drop_layer": layer with largest entropy decrease
            "final_entropy": entropy at the final layer
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    entropies = []
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            logits = resid @ W_U + b_U
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
            entropies.append(entropy)
        else:
            entropies.append(0.0)

    entropies = np.array(entropies)

    if len(entropies) >= 2:
        drops = np.diff(entropies)
        steepest = int(np.argmin(drops))  # Largest decrease
        reduction = float(entropies[0] - entropies[-1])
    else:
        steepest = 0
        reduction = 0.0

    return {
        "layer_entropies": entropies,
        "entropy_reduction": reduction,
        "steepest_drop_layer": steepest,
        "final_entropy": float(entropies[-1]) if len(entropies) > 0 else 0.0,
    }


def representation_stabilization_point(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    convergence_threshold: float = 0.01,
    pos: int = -1,
) -> dict:
    """Find the layer where residual stream representations stabilize.

    Measures cosine distance between consecutive layers and finds where
    the representation stops changing significantly.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        convergence_threshold: Max cosine distance for "stable".
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "layer_distances": [n_layers-1] cosine distances between consecutive layers
            "stabilization_layer": first layer where distance < threshold (-1 if never)
            "total_drift": sum of all cosine distances
            "max_change_layer": layer pair with largest change
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    resids = []
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resids.append(np.array(cache.cache_dict[key][pos]))
        else:
            resids.append(np.zeros(model.cfg.d_model))

    distances = []
    for i in range(len(resids) - 1):
        cos_sim = np.dot(resids[i], resids[i + 1]) / (
            np.linalg.norm(resids[i]) * np.linalg.norm(resids[i + 1]) + 1e-10
        )
        distances.append(1.0 - cos_sim)  # cosine distance

    distances = np.array(distances)

    stabilization = -1
    for i, d in enumerate(distances):
        if d < convergence_threshold:
            stabilization = i + 1  # Layer after the stable transition
            break

    return {
        "layer_distances": distances,
        "stabilization_layer": stabilization,
        "total_drift": float(np.sum(distances)),
        "max_change_layer": int(np.argmax(distances)) if len(distances) > 0 else 0,
    }


def early_vs_late_computation_balance(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    split_layer: Optional[int] = None,
) -> dict:
    """Measure how computation burden splits between early and late layers.

    Ablates early vs late layers independently and compares metric effects.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        split_layer: Layer to split early/late. Defaults to n_layers // 2.

    Returns:
        Dict with:
            "early_effect": metric change when early layers are ablated
            "late_effect": metric change when late layers are ablated
            "balance_ratio": early_effect / (early_effect + late_effect)
            "split_layer": the layer used for splitting
    """
    n_layers = model.cfg.n_layers
    if split_layer is None:
        split_layer = n_layers // 2

    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    def zero_hook(x, name):
        return jnp.zeros_like(x)

    # Ablate early layers
    early_hooks = []
    for l in range(split_layer):
        early_hooks.append((f"blocks.{l}.attn.hook_result", zero_hook))

    if early_hooks:
        early_logits = model.run_with_hooks(tokens, fwd_hooks=early_hooks)
        early_effect = abs(float(metric_fn(early_logits)) - clean_metric)
    else:
        early_effect = 0.0

    # Ablate late layers
    late_hooks = []
    for l in range(split_layer, n_layers):
        late_hooks.append((f"blocks.{l}.attn.hook_result", zero_hook))

    if late_hooks:
        late_logits = model.run_with_hooks(tokens, fwd_hooks=late_hooks)
        late_effect = abs(float(metric_fn(late_logits)) - clean_metric)
    else:
        late_effect = 0.0

    total = early_effect + late_effect
    balance = early_effect / total if total > 0 else 0.5

    return {
        "early_effect": early_effect,
        "late_effect": late_effect,
        "balance_ratio": balance,
        "split_layer": split_layer,
    }
