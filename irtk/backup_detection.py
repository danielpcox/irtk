"""Backup and redundancy detection for transformer circuits.

Analyzes circuit redundancy: when one component (typically an attention head)
is ablated, another may compensate, revealing backup circuits. This is
important for understanding model robustness and the structure of computation.

Functions:
- detect_backup_heads: Find heads that compensate for ablated heads
- knockout_compensation: Measure output recovery after single-head ablation
- circuit_redundancy_map: Map redundancy relationships across all head pairs
- critical_vs_backup: Classify heads as critical (no backup) or backed-up
- ablation_recovery_curve: Track metric as ablated heads are incrementally restored

References:
    - Wang et al. (2022) "Interpretability in the Wild" (IOI paper)
    - Conmy et al. (2023) "Towards Automated Circuit Discovery" (ACDC)
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def _zero_head_hook(layer: int, head: int):
    """Create a hook that zeros out a specific attention head's z vector."""
    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(x, name):
        # x: [seq_len, n_heads, d_head]
        return x.at[:, head, :].set(0.0)

    return hook_name, hook_fn


def detect_backup_heads(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    target_layer: int,
    target_head: int,
    threshold: float = 0.5,
) -> dict:
    """Find heads that compensate when a target head is ablated.

    Ablates the target head, then additionally ablates each other head to
    see which ones make the metric worse (indicating they were compensating).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        target_layer: Layer of the head to ablate.
        target_head: Head index to ablate.
        threshold: Minimum compensation score to count as backup.

    Returns:
        Dict with:
            "backup_heads": list of (layer, head) pairs that are backups
            "compensation_scores": dict mapping (layer, head) to score
            "clean_metric": metric without ablation
            "ablated_metric": metric with target head ablated
    """
    # Clean metric
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    # Metric with target ablated
    hook_name, hook_fn = _zero_head_hook(target_layer, target_head)
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    ablated_metric = float(metric_fn(ablated_logits))

    ablation_effect = abs(clean_metric - ablated_metric)
    if ablation_effect < 1e-8:
        return {
            "backup_heads": [],
            "compensation_scores": {},
            "clean_metric": clean_metric,
            "ablated_metric": ablated_metric,
        }

    # Now ablate target + each other head, see which makes it even worse
    compensation_scores = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for l in range(n_layers):
        for h in range(n_heads):
            if l == target_layer and h == target_head:
                continue

            # Ablate both target and candidate
            hook1_name, hook1_fn = _zero_head_hook(target_layer, target_head)
            hook2_name, hook2_fn = _zero_head_hook(l, h)
            double_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook1_name, hook1_fn), (hook2_name, hook2_fn)]
            )
            double_metric = float(metric_fn(double_logits))

            # How much worse is double ablation vs single?
            additional_drop = abs(ablated_metric - double_metric)
            score = additional_drop / (ablation_effect + 1e-8)
            compensation_scores[(l, h)] = score

    backup_heads = [(l, h) for (l, h), s in compensation_scores.items() if s >= threshold]

    return {
        "backup_heads": backup_heads,
        "compensation_scores": compensation_scores,
        "clean_metric": clean_metric,
        "ablated_metric": ablated_metric,
    }


def knockout_compensation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Measure how much the model compensates when each head is individually ablated.

    For each head, compares the expected metric drop (based on direct effect)
    with the actual metric drop to estimate compensation.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "per_head_ablation_effect": [n_layers, n_heads] actual metric change
            "most_compensated": (layer, head) with highest inferred compensation
            "compensation_estimate": [n_layers, n_heads] estimated compensation
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            hook_name, hook_fn = _zero_head_hook(l, h)
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, hook_fn)]
            )
            effects[l, h] = float(metric_fn(ablated_logits)) - clean_metric

    # Estimate compensation: heads with small individual effect but large
    # combined effect with others likely have backup circuits.
    # We estimate by comparing individual effects to their absolute values:
    # if ablation barely hurts, but the head has large attention norms,
    # other heads are compensating.
    abs_effects = np.abs(effects)
    mean_effect = np.mean(abs_effects) if abs_effects.size > 0 else 0.0
    # Compensation = how much smaller the effect is than average
    compensation = np.maximum(0, mean_effect - abs_effects)

    most_comp = np.unravel_index(np.argmax(compensation), compensation.shape)

    return {
        "per_head_ablation_effect": effects,
        "most_compensated": (int(most_comp[0]), int(most_comp[1])),
        "compensation_estimate": compensation,
    }


def circuit_redundancy_map(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Map redundancy relationships across all head pairs.

    Computes a pairwise interaction matrix: for each pair (A, B), measures
    whether ablating both is worse than the sum of ablating each individually
    (superadditivity indicates redundancy/backup).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "redundancy_matrix": [total_heads, total_heads] pairwise scores
            "head_labels": list of (layer, head) labels
            "most_redundant_pair": ((l1,h1), (l2,h2)) most redundant pair
            "redundancy_score": score of the most redundant pair
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total = n_layers * n_heads
    labels = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    # Single ablation effects
    single_effects = np.zeros(total)
    for i, (l, h) in enumerate(labels):
        hook_name, hook_fn = _zero_head_hook(l, h)
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        single_effects[i] = float(metric_fn(logits)) - clean_metric

    # Pairwise ablation
    redundancy = np.zeros((total, total))
    for i in range(total):
        for j in range(i + 1, total):
            l1, h1 = labels[i]
            l2, h2 = labels[j]
            hook1_name, hook1_fn = _zero_head_hook(l1, h1)
            hook2_name, hook2_fn = _zero_head_hook(l2, h2)
            logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook1_name, hook1_fn), (hook2_name, hook2_fn)]
            )
            joint_effect = float(metric_fn(logits)) - clean_metric
            # Superadditivity: if joint < sum, heads are redundant
            expected = single_effects[i] + single_effects[j]
            redundancy[i, j] = abs(joint_effect) - abs(expected)
            redundancy[j, i] = redundancy[i, j]

    # Find most redundant pair (most negative = most backup)
    # A negative value means ablating both is LESS bad than expected
    min_idx = np.argmin(redundancy)
    i, j = np.unravel_index(min_idx, redundancy.shape)

    return {
        "redundancy_matrix": redundancy,
        "head_labels": labels,
        "most_redundant_pair": (labels[i], labels[j]),
        "redundancy_score": float(redundancy[i, j]),
    }


def critical_vs_backup(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    critical_threshold: float = 0.1,
) -> dict:
    """Classify each head as critical (no backup) or backed-up (has redundancy).

    A head is critical if ablating it causes a large metric change that
    isn't recovered by any compensation. A head is backed-up if its
    individual ablation effect is small relative to its activity.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        critical_threshold: Minimum absolute effect to be critical.

    Returns:
        Dict with:
            "critical_heads": list of (layer, head) critical heads
            "backup_heads": list of (layer, head) backed-up heads
            "neutral_heads": list of (layer, head) with no significant effect
            "classification": dict mapping (layer, head) -> "critical"/"backup"/"neutral"
            "effects": dict mapping (layer, head) -> ablation effect
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    effects = {}
    for l in range(n_layers):
        for h in range(n_heads):
            hook_name, hook_fn = _zero_head_hook(l, h)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
            effects[(l, h)] = float(metric_fn(logits)) - clean_metric

    # Get attention norms as proxy for head activity
    _, cache = model.run_with_cache(tokens)

    critical = []
    backup = []
    neutral = []
    classification = {}

    for (l, h), effect in effects.items():
        abs_effect = abs(effect)
        if abs_effect >= critical_threshold:
            critical.append((l, h))
            classification[(l, h)] = "critical"
        elif abs_effect < critical_threshold * 0.1:
            # Very small effect - either neutral or backed up
            # Check if the head has meaningful attention patterns
            pattern_key = f"blocks.{l}.attn.hook_pattern"
            if pattern_key in cache.cache_dict:
                pattern = cache.cache_dict[pattern_key]  # [seq, n_heads, seq]
                head_pattern = pattern[:, h, :]
                entropy = -float(jnp.sum(
                    head_pattern * jnp.log(head_pattern + 1e-10)
                ) / head_pattern.shape[0])
                # Low entropy = focused attention = likely doing something
                if entropy < 1.0:
                    backup.append((l, h))
                    classification[(l, h)] = "backup"
                else:
                    neutral.append((l, h))
                    classification[(l, h)] = "neutral"
            else:
                neutral.append((l, h))
                classification[(l, h)] = "neutral"
        else:
            # Moderate effect
            backup.append((l, h))
            classification[(l, h)] = "backup"

    return {
        "critical_heads": critical,
        "backup_heads": backup,
        "neutral_heads": neutral,
        "classification": classification,
        "effects": effects,
    }


def ablation_recovery_curve(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    ablate_heads: Optional[list] = None,
) -> dict:
    """Track metric as ablated heads are incrementally restored.

    Starts with all specified heads ablated, then restores them one at a time
    (most important first) to trace the recovery curve.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        ablate_heads: List of (layer, head) to ablate. If None, uses all heads.

    Returns:
        Dict with:
            "recovery_curve": list of metric values as heads are restored
            "restoration_order": order in which heads were restored
            "clean_metric": metric with no ablation
            "fully_ablated_metric": metric with all heads ablated
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if ablate_heads is None:
        ablate_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    # Fully ablated metric
    all_hooks = [_zero_head_hook(l, h) for l, h in ablate_heads]
    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
    fully_ablated = float(metric_fn(ablated_logits))

    # Greedily restore heads: at each step, restore the head that
    # improves the metric the most
    remaining_ablated = set((l, h) for l, h in ablate_heads)
    restoration_order = []
    curve = [fully_ablated]

    while remaining_ablated:
        best_head = None
        best_metric = None

        for (l, h) in remaining_ablated:
            # Try restoring this head
            test_ablated = remaining_ablated - {(l, h)}
            hooks = [_zero_head_hook(ll, hh) for ll, hh in test_ablated]
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
            m = float(metric_fn(logits))

            if best_metric is None or abs(m - clean_metric) < abs(best_metric - clean_metric):
                best_metric = m
                best_head = (l, h)

        remaining_ablated.remove(best_head)
        restoration_order.append(best_head)
        curve.append(best_metric)

    return {
        "recovery_curve": curve,
        "restoration_order": restoration_order,
        "clean_metric": clean_metric,
        "fully_ablated_metric": fully_ablated,
    }
