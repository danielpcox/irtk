"""Systematic ablation study framework.

Structured tools for running ablation experiments across component
classes, aggregating results, and computing sensitivity statistics:
- layer_by_layer_ablation: Ablate each layer in turn
- head_importance_matrix: Per-head ablation effects
- position_sensitivity: Per-position importance via embedding ablation
- double_ablation_interaction: Pairwise joint ablation effects
- ablation_summary: Aggregate results into actionable findings
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_by_layer_ablation(
    model: HookedTransformer,
    token_sequences: list,
    metric_fn: Callable[[jnp.ndarray], float],
    component: str = "attn",
    ablation_type: str = "zero",
) -> dict:
    """Ablate each layer's component in turn across a dataset.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        component: "attn" or "mlp".
        ablation_type: "zero" or "mean".

    Returns:
        Dict with:
        - "effects": [n_layers] mean ablation effect per layer
        - "std": [n_layers] standard deviation across prompts
        - "clean_metrics": [n_prompts] baseline metrics
        - "per_prompt": [n_layers, n_prompts] effects per prompt
    """
    n_layers = model.cfg.n_layers
    hook_suffix = "hook_attn_out" if component == "attn" else "hook_mlp_out"

    clean_metrics = []
    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        clean_metrics.append(metric_fn(model(tokens)))
    clean_metrics = np.array(clean_metrics)

    per_prompt = np.zeros((n_layers, len(token_sequences)))

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.{hook_suffix}"

        for pi, tokens in enumerate(token_sequences):
            tokens = jnp.array(tokens)

            if ablation_type == "mean":
                _, cache = model.run_with_cache(tokens)
                if hook_name in cache.cache_dict:
                    mean_act = jnp.mean(cache.cache_dict[hook_name], axis=0, keepdims=True)
                    replacement = mean_act * jnp.ones_like(cache.cache_dict[hook_name])
                    def hook_fn(x, name, _r=replacement):
                        return _r
                else:
                    def hook_fn(x, name):
                        return x
            else:
                def hook_fn(x, name):
                    return jnp.zeros_like(x)

            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
            per_prompt[layer, pi] = metric_fn(logits) - clean_metrics[pi]

    return {
        "effects": np.mean(per_prompt, axis=1),
        "std": np.std(per_prompt, axis=1),
        "clean_metrics": clean_metrics,
        "per_prompt": per_prompt,
    }


def head_importance_matrix(
    model: HookedTransformer,
    token_sequences: list,
    metric_fn: Callable[[jnp.ndarray], float],
    ablation_type: str = "zero",
) -> dict:
    """Per-head ablation effect matrix across a dataset.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        ablation_type: "zero" or "mean".

    Returns:
        Dict with:
        - "matrix": [n_layers, n_heads] mean ablation effect
        - "std_matrix": [n_layers, n_heads] std across prompts
        - "clean_mean": mean clean metric
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_metrics = []
    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        clean_metrics.append(metric_fn(model(tokens)))
    clean_mean = float(np.mean(clean_metrics))

    matrix = np.zeros((n_layers, n_heads))
    std_matrix = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"

        for head in range(n_heads):
            effects = []
            for pi, tokens in enumerate(token_sequences):
                tokens = jnp.array(tokens)

                def ablate_head(x, name, _h=head):
                    return x.at[:, _h, :].set(0.0)

                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate_head)])
                effects.append(metric_fn(logits) - clean_metrics[pi])

            matrix[layer, head] = np.mean(effects)
            std_matrix[layer, head] = np.std(effects)

    return {
        "matrix": matrix,
        "std_matrix": std_matrix,
        "clean_mean": clean_mean,
    }


def position_sensitivity(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    ablation_type: str = "zero",
) -> dict:
    """Measure importance of each input position via embedding ablation.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        ablation_type: "zero" (zero embedding) or "mean" (mean embedding).

    Returns:
        Dict with:
        - "effects": [seq_len] effect of ablating each position
        - "clean_metric": baseline metric
        - "most_important_pos": position with largest |effect|
    """
    tokens = jnp.array(tokens)
    clean_metric = metric_fn(model(tokens))
    seq_len = len(tokens)

    effects = np.zeros(seq_len)

    for pos in range(seq_len):
        if ablation_type == "mean":
            _, cache = model.run_with_cache(tokens)
            embed = cache.cache_dict.get("hook_embed")
            if embed is not None:
                mean_embed = jnp.mean(embed, axis=0)
                def hook_fn(x, name, _p=pos, _m=mean_embed):
                    return x.at[_p].set(_m)
            else:
                def hook_fn(x, name):
                    return x
        else:
            def hook_fn(x, name, _p=pos):
                return x.at[_p].set(0.0)

        logits = model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", hook_fn)])
        effects[pos] = metric_fn(logits) - clean_metric

    most_important = int(np.argmax(np.abs(effects)))

    return {
        "effects": effects,
        "clean_metric": float(clean_metric),
        "most_important_pos": most_important,
    }


def double_ablation_interaction(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    hooks_a: list[str],
    hooks_b: list[str],
) -> dict:
    """Compute joint ablation effects for pairs of hooks.

    For each (a, b) pair, computes:
    - effect_a: ablating a alone
    - effect_b: ablating b alone
    - effect_both: ablating both
    - interaction: effect_both - effect_a - effect_b

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function(logits) -> float.
        hooks_a: First set of hooks.
        hooks_b: Second set of hooks.

    Returns:
        Dict with:
        - "interaction_matrix": [len(a), len(b)] interaction effects
        - "effects_a": [len(a)] individual effects
        - "effects_b": [len(b)] individual effects
        - "joint_effects": [len(a), len(b)] joint ablation effects
    """
    tokens = jnp.array(tokens)
    clean_metric = metric_fn(model(tokens))

    def zero_hook(x, name):
        return jnp.zeros_like(x)

    # Individual effects for A
    effects_a = np.zeros(len(hooks_a))
    for i, hook in enumerate(hooks_a):
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook, zero_hook)])
        effects_a[i] = metric_fn(logits) - clean_metric

    # Individual effects for B
    effects_b = np.zeros(len(hooks_b))
    for j, hook in enumerate(hooks_b):
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook, zero_hook)])
        effects_b[j] = metric_fn(logits) - clean_metric

    # Joint effects
    joint = np.zeros((len(hooks_a), len(hooks_b)))
    for i, ha in enumerate(hooks_a):
        for j, hb in enumerate(hooks_b):
            if ha == hb:
                joint[i, j] = effects_a[i]
            else:
                logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(ha, zero_hook), (hb, zero_hook)]
                )
                joint[i, j] = metric_fn(logits) - clean_metric

    # Interaction = joint - individual_a - individual_b
    interaction = joint - effects_a.reshape(-1, 1) - effects_b.reshape(1, -1)

    return {
        "interaction_matrix": interaction,
        "effects_a": effects_a,
        "effects_b": effects_b,
        "joint_effects": joint,
    }


def ablation_summary(
    effects: np.ndarray,
    labels: Optional[list[str]] = None,
    top_k: int = 10,
) -> dict:
    """Aggregate raw ablation effects into a summary.

    Args:
        effects: Array of ablation effects (any shape, will be flattened).
        labels: Optional labels for each effect.
        top_k: Number of top components to highlight.

    Returns:
        Dict with:
        - "top_components": list of (label, effect) sorted by |effect|
        - "mean_effect": mean absolute effect
        - "max_effect": maximum absolute effect
        - "gini_coefficient": inequality of importance distribution
        - "n_important": number with |effect| > mean
    """
    flat = np.array(effects).ravel()
    abs_flat = np.abs(flat)

    if labels is None:
        labels = [str(i) for i in range(len(flat))]
    elif len(labels) != len(flat):
        labels = [str(i) for i in range(len(flat))]

    # Sort by absolute effect
    sorted_idx = np.argsort(abs_flat)[::-1]
    top = [(labels[i], float(flat[i])) for i in sorted_idx[:top_k]]

    mean_effect = float(np.mean(abs_flat))
    max_effect = float(np.max(abs_flat)) if len(abs_flat) > 0 else 0.0
    n_important = int(np.sum(abs_flat > mean_effect))

    # Gini coefficient
    if len(abs_flat) > 1 and abs_flat.sum() > 1e-10:
        sorted_vals = np.sort(abs_flat)
        n = len(sorted_vals)
        cumulative = np.cumsum(sorted_vals)
        gini = float((2 * np.sum((np.arange(1, n + 1) * sorted_vals))) / (n * cumulative[-1]) - (n + 1) / n)
        gini = max(0.0, min(1.0, gini))
    else:
        gini = 0.0

    return {
        "top_components": top,
        "mean_effect": mean_effect,
        "max_effect": max_effect,
        "gini_coefficient": gini,
        "n_important": n_important,
    }
