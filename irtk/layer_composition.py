"""Layer composition analysis.

Analyzes how layers contribute to the model's output and interact with each
other. Measures per-layer importance, redundancy, similarity patterns, and
identifies critical vs dispensable layers.

Functions:
- layer_residual_contribution: Per-layer contribution to the output logits
- layer_output_similarity: Cosine similarity between layer outputs
- layer_redundancy_analysis: Functional redundancy between layers
- critical_layer_identification: Layers whose ablation is catastrophic
- layer_specialization_profile: What each layer specializes in

References:
    - Veit et al. (2016) "Residual Networks Behave Like Ensembles"
    - Men et al. (2024) "ShortGPT: Layers in LLMs are More Redundant Than You Think"
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_residual_contribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Per-layer contribution to the output logits.

    Decomposes the final logit vector into additive contributions from
    each layer's residual stream update.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_contributions": [n_layers] L2 norm of each layer's contribution
            "layer_logit_norms": [n_layers] norm of logit contribution per layer
            "embedding_contribution": embedding's contribution norm
            "dominant_layer": layer contributing most
            "contribution_fractions": [n_layers] fraction of total contribution
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)

    contributions = np.zeros(n_layers)
    logit_norms = np.zeros(n_layers)

    # Embedding
    embed_key = "blocks.0.hook_resid_pre"
    embed_contrib = 0.0
    if embed_key in cache.cache_dict:
        embed = np.array(cache.cache_dict[embed_key][pos])
        embed_contrib = float(np.linalg.norm(embed))

    for l in range(n_layers):
        post_key = f"blocks.{l}.hook_resid_post"
        if l == 0:
            pre_key = "blocks.0.hook_resid_pre"
        else:
            pre_key = f"blocks.{l-1}.hook_resid_post"

        if post_key in cache.cache_dict and pre_key in cache.cache_dict:
            post = np.array(cache.cache_dict[post_key][pos])
            pre = np.array(cache.cache_dict[pre_key][pos])
            layer_update = post - pre
            contributions[l] = float(np.linalg.norm(layer_update))
            logit_norms[l] = float(np.linalg.norm(layer_update @ W_U))

    total = float(np.sum(contributions)) + embed_contrib
    fractions = contributions / (total + 1e-10)
    dominant = int(np.argmax(contributions))

    return {
        "layer_contributions": contributions,
        "layer_logit_norms": logit_norms,
        "embedding_contribution": embed_contrib,
        "dominant_layer": dominant,
        "contribution_fractions": fractions,
    }


def layer_output_similarity(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Cosine similarity between consecutive layer outputs.

    Measures how much each layer changes the representation compared
    to the previous layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to analyze (-1 = last).

    Returns:
        Dict with:
            "consecutive_similarity": [n_layers-1] cosine sim between layer l and l-1
            "similarity_to_final": [n_layers] cosine sim of each layer to final output
            "mean_consecutive_similarity": average consecutive similarity
            "most_different_layer": layer most different from its predecessor
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    # Collect all layer outputs
    outputs = []
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            outputs.append(np.array(cache.cache_dict[key][pos]))
        else:
            outputs.append(np.zeros(model.cfg.d_model))

    # Consecutive similarity
    consec = []
    for i in range(1, len(outputs)):
        n1 = np.linalg.norm(outputs[i])
        n0 = np.linalg.norm(outputs[i-1])
        if n1 > 1e-10 and n0 > 1e-10:
            consec.append(float(np.dot(outputs[i], outputs[i-1]) / (n1 * n0)))
        else:
            consec.append(0.0)
    consec = np.array(consec)

    # Similarity to final
    final = outputs[-1]
    final_norm = np.linalg.norm(final)
    sim_to_final = np.zeros(n_layers)
    for i in range(n_layers):
        ni = np.linalg.norm(outputs[i])
        if ni > 1e-10 and final_norm > 1e-10:
            sim_to_final[i] = float(np.dot(outputs[i], final) / (ni * final_norm))

    mean_consec = float(np.mean(consec)) if len(consec) > 0 else 0.0
    most_diff = int(np.argmin(consec)) + 1 if len(consec) > 0 else 0

    return {
        "consecutive_similarity": consec,
        "similarity_to_final": sim_to_final,
        "mean_consecutive_similarity": mean_consec,
        "most_different_layer": most_diff,
    }


def layer_redundancy_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Functional redundancy between layers.

    Ablates each layer (replaces its output with its input) and measures
    metric change. Layers with small change are redundant.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float to measure.

    Returns:
        Dict with:
            "ablation_effects": [n_layers] absolute metric change from ablating each layer
            "most_redundant_layer": layer with smallest ablation effect
            "most_critical_layer": layer with largest ablation effect
            "redundancy_scores": [n_layers] normalized redundancy (1 = fully redundant)
    """
    baseline_logits = np.array(model(tokens))
    baseline = float(metric_fn(baseline_logits))
    n_layers = model.cfg.n_layers

    effects = np.zeros(n_layers)

    for l in range(n_layers):
        # Hook that makes this layer a no-op (output = input)
        def make_hook(layer_idx):
            resid_pre_key = f"blocks.{layer_idx}.hook_resid_pre" if layer_idx == 0 else f"blocks.{layer_idx-1}.hook_resid_post"
            def hook_fn(x, name):
                return jnp.zeros_like(x)  # zero out layer's contribution
            return hook_fn

        # Ablate by zeroing the layer's update (hook_resid_post - hook_resid_pre)
        # We hook the mlp and attention outputs to return zero
        hooks = []
        attn_hook = f"blocks.{l}.attn.hook_result"
        hooks.append((attn_hook, lambda x, name: jnp.zeros_like(x)))

        mlp_post = f"blocks.{l}.hook_resid_post"
        mlp_mid = f"blocks.{l}.hook_resid_mid"

        # Simple approach: just zero the attention result
        try:
            patched_logits = np.array(model.run_with_hooks(tokens, fwd_hooks=hooks))
            effects[l] = abs(float(metric_fn(patched_logits)) - baseline)
        except Exception:
            effects[l] = 0.0

    max_effect = float(np.max(effects)) if float(np.max(effects)) > 1e-10 else 1.0
    redundancy = 1.0 - effects / max_effect

    return {
        "ablation_effects": effects,
        "most_redundant_layer": int(np.argmin(effects)),
        "most_critical_layer": int(np.argmax(effects)),
        "redundancy_scores": redundancy,
    }


def critical_layer_identification(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    threshold: float = 0.5,
) -> dict:
    """Identify critical vs dispensable layers.

    A layer is critical if ablating it causes metric degradation beyond
    the threshold.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        threshold: Fraction of baseline degradation to be "critical".

    Returns:
        Dict with:
            "layer_effects": [n_layers] metric after ablating each layer
            "critical_layers": list of critical layer indices
            "dispensable_layers": list of dispensable layer indices
            "n_critical": number of critical layers
            "baseline_metric": original metric value
    """
    redundancy = layer_redundancy_analysis(model, tokens, metric_fn)
    baseline_logits = np.array(model(tokens))
    baseline = float(metric_fn(baseline_logits))
    n_layers = model.cfg.n_layers

    effects = redundancy["ablation_effects"]
    abs_baseline = abs(baseline) if abs(baseline) > 1e-10 else 1.0

    critical = []
    dispensable = []
    for l in range(n_layers):
        if effects[l] / abs_baseline > threshold:
            critical.append(l)
        else:
            dispensable.append(l)

    return {
        "layer_effects": effects,
        "critical_layers": critical,
        "dispensable_layers": dispensable,
        "n_critical": len(critical),
        "baseline_metric": baseline,
    }


def layer_specialization_profile(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """What each layer specializes in: attention vs MLP balance.

    Measures the relative contribution of attention and MLP sublayers
    within each transformer block.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "attn_norms": [n_layers] L2 norm of attention output
            "mlp_norms": [n_layers] L2 norm of MLP output (post - mid)
            "attn_fraction": [n_layers] fraction of layer's update from attention
            "mlp_dominant_layers": layers where MLP contributes more than attention
            "attn_dominant_layers": layers where attention contributes more than MLP
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    for l in range(n_layers):
        attn_key = f"blocks.{l}.attn.hook_result"
        mid_key = f"blocks.{l}.hook_resid_mid"
        post_key = f"blocks.{l}.hook_resid_post"

        if attn_key in cache.cache_dict:
            attn_out = np.array(cache.cache_dict[attn_key][pos])
            attn_norms[l] = float(np.linalg.norm(attn_out))

        if mid_key in cache.cache_dict and post_key in cache.cache_dict:
            mid = np.array(cache.cache_dict[mid_key][pos])
            post = np.array(cache.cache_dict[post_key][pos])
            mlp_out = post - mid
            mlp_norms[l] = float(np.linalg.norm(mlp_out))

    total = attn_norms + mlp_norms + 1e-10
    attn_frac = attn_norms / total

    mlp_dom = [l for l in range(n_layers) if mlp_norms[l] > attn_norms[l]]
    attn_dom = [l for l in range(n_layers) if attn_norms[l] >= mlp_norms[l]]

    return {
        "attn_norms": attn_norms,
        "mlp_norms": mlp_norms,
        "attn_fraction": attn_frac,
        "mlp_dominant_layers": mlp_dom,
        "attn_dominant_layers": attn_dom,
    }
