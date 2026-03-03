"""Circuit evaluation metrics for assessing discovered circuits.

Tools for scientifically evaluating whether a discovered circuit is a
faithful, complete, and minimal explanation of model behavior:
- faithfulness_score: Does the circuit reproduce the full model's behavior?
- completeness_score: Does ablating the circuit destroy the behavior?
- minimality_check: Can any component be removed without loss?
- circuit_iou: Overlap between two circuits
- evaluate_circuit: All metrics in one call
"""

from typing import Callable

import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def _ablate_heads_outside_circuit(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    circuit_heads: list[tuple[int, int]],
    method: str = "zero",
) -> jnp.ndarray:
    """Run model with all heads NOT in circuit ablated.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        circuit_heads: List of (layer, head) tuples that form the circuit.
        method: "zero" or "mean".

    Returns:
        Logits from the ablated run.
    """
    circuit_set = set((l, h) for l, h in circuit_heads)

    _, cache = model.run_with_cache(tokens)

    fwd_hooks = []
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        z = cache[hook_name]

        heads_to_ablate = []
        for head in range(model.cfg.n_heads):
            if (layer, head) not in circuit_set:
                heads_to_ablate.append(head)

        if not heads_to_ablate:
            continue

        if method == "zero":
            def ablate_hook(x, name, _heads=heads_to_ablate):
                for h in _heads:
                    x = x.at[:, h, :].set(0.0)
                return x
        else:
            head_means = {}
            for h in heads_to_ablate:
                head_means[h] = jnp.mean(z[:, h, :], axis=0)

            def ablate_hook(x, name, _hm=head_means):
                for h, m in _hm.items():
                    x = x.at[:, h, :].set(m)
                return x

        fwd_hooks.append((hook_name, ablate_hook))

    return model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)


def _ablate_heads_inside_circuit(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    circuit_heads: list[tuple[int, int]],
    method: str = "zero",
) -> jnp.ndarray:
    """Run model with all heads IN the circuit ablated.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        circuit_heads: List of (layer, head) tuples that form the circuit.
        method: "zero" or "mean".

    Returns:
        Logits from the ablated run.
    """
    circuit_set = set((l, h) for l, h in circuit_heads)

    _, cache = model.run_with_cache(tokens)

    fwd_hooks = []
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        z = cache[hook_name]

        heads_to_ablate = []
        for head in range(model.cfg.n_heads):
            if (layer, head) in circuit_set:
                heads_to_ablate.append(head)

        if not heads_to_ablate:
            continue

        if method == "zero":
            def ablate_hook(x, name, _heads=heads_to_ablate):
                for h in _heads:
                    x = x.at[:, h, :].set(0.0)
                return x
        else:
            head_means = {}
            for h in heads_to_ablate:
                head_means[h] = jnp.mean(z[:, h, :], axis=0)

            def ablate_hook(x, name, _hm=head_means):
                for h, m in _hm.items():
                    x = x.at[:, h, :].set(m)
                return x

        fwd_hooks.append((hook_name, ablate_hook))

    return model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)


def faithfulness_score(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    method: str = "zero",
) -> float:
    """Measure how faithfully the circuit reproduces the full model's behavior.

    Ablates all heads NOT in the circuit and measures how much of the
    original metric is preserved. A score of 1.0 means perfect recovery.

    faithfulness = 1 - |metric(full) - metric(circuit_only)| / |metric(full) - metric(all_ablated)|

    Args:
        model: HookedTransformer.
        circuit_heads: List of (layer, head) tuples.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        method: Ablation method ("zero" or "mean").

    Returns:
        Faithfulness score in [0, 1]. Higher is better.
    """
    full_values = []
    circuit_values = []
    ablated_values = []

    all_heads = [(l, h) for l in range(model.cfg.n_layers)
                 for h in range(model.cfg.n_heads)]

    for tokens in token_sequences:
        # Full model
        logits = model(tokens)
        full_values.append(metric_fn(logits))

        # Circuit only (ablate everything outside)
        logits = _ablate_heads_outside_circuit(model, tokens, circuit_heads, method)
        circuit_values.append(metric_fn(logits))

        # All ablated (no circuit)
        logits = _ablate_heads_outside_circuit(model, tokens, [], method)
        ablated_values.append(metric_fn(logits))

    full_mean = np.mean(full_values)
    circuit_mean = np.mean(circuit_values)
    ablated_mean = np.mean(ablated_values)

    denominator = abs(full_mean - ablated_mean)
    if denominator < 1e-10:
        return 1.0

    score = 1.0 - abs(full_mean - circuit_mean) / denominator
    return float(np.clip(score, 0.0, 1.0))


def completeness_score(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    method: str = "zero",
) -> float:
    """Measure whether the circuit is sufficient for the behavior.

    Ablates all heads IN the circuit and measures how much the metric
    drops. A score of 1.0 means the circuit is fully responsible.

    completeness = |metric(full) - metric(circuit_ablated)| / |metric(full) - metric(all_ablated)|

    Args:
        model: HookedTransformer.
        circuit_heads: List of (layer, head) tuples.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        method: Ablation method ("zero" or "mean").

    Returns:
        Completeness score in [0, 1]. Higher is better.
    """
    full_values = []
    circuit_ablated_values = []
    all_ablated_values = []

    for tokens in token_sequences:
        logits = model(tokens)
        full_values.append(metric_fn(logits))

        logits = _ablate_heads_inside_circuit(model, tokens, circuit_heads, method)
        circuit_ablated_values.append(metric_fn(logits))

        logits = _ablate_heads_outside_circuit(model, tokens, [], method)
        all_ablated_values.append(metric_fn(logits))

    full_mean = np.mean(full_values)
    circuit_ablated_mean = np.mean(circuit_ablated_values)
    all_ablated_mean = np.mean(all_ablated_values)

    denominator = abs(full_mean - all_ablated_mean)
    if denominator < 1e-10:
        return 1.0

    score = abs(full_mean - circuit_ablated_mean) / denominator
    return float(np.clip(score, 0.0, 1.0))


def minimality_check(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    threshold: float = 0.05,
    method: str = "zero",
) -> list[dict]:
    """Check which circuit components can be removed without significant loss.

    For each head in the circuit, removes it and measures the metric change.
    Heads that can be removed (change < threshold) are flagged as redundant.

    Args:
        model: HookedTransformer.
        circuit_heads: List of (layer, head) tuples.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        threshold: Maximum relative change to consider a head redundant.
        method: Ablation method ("zero" or "mean").

    Returns:
        List of dicts, one per circuit head, with:
        - "layer", "head": Component identity
        - "metric_without": Metric with this head also ablated
        - "metric_change": Absolute change vs circuit-only metric
        - "relative_change": Change relative to circuit-only metric
        - "redundant": Whether this head can be removed
    """
    # Circuit-only baseline
    circuit_values = []
    for tokens in token_sequences:
        logits = _ablate_heads_outside_circuit(model, tokens, circuit_heads, method)
        circuit_values.append(metric_fn(logits))
    circuit_baseline = np.mean(circuit_values)

    results = []
    for i, (layer, head) in enumerate(circuit_heads):
        # Remove this head from the circuit
        reduced = [h for j, h in enumerate(circuit_heads) if j != i]

        reduced_values = []
        for tokens in token_sequences:
            logits = _ablate_heads_outside_circuit(model, tokens, reduced, method)
            reduced_values.append(metric_fn(logits))
        reduced_metric = np.mean(reduced_values)

        change = abs(circuit_baseline - reduced_metric)
        rel_change = change / abs(circuit_baseline) if abs(circuit_baseline) > 1e-10 else 0.0

        results.append({
            "layer": layer,
            "head": head,
            "metric_without": float(reduced_metric),
            "metric_change": float(change),
            "relative_change": float(rel_change),
            "redundant": rel_change < threshold,
        })

    return results


def circuit_iou(
    circuit_a: list[tuple[int, int]],
    circuit_b: list[tuple[int, int]],
) -> float:
    """Compute intersection-over-union between two circuits.

    Args:
        circuit_a: First circuit as list of (layer, head) tuples.
        circuit_b: Second circuit as list of (layer, head) tuples.

    Returns:
        IoU score in [0, 1]. 1.0 means identical circuits.
    """
    set_a = set(circuit_a)
    set_b = set(circuit_b)

    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def evaluate_circuit(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    method: str = "zero",
    minimality_threshold: float = 0.05,
) -> dict:
    """Comprehensive circuit evaluation with all metrics.

    Args:
        model: HookedTransformer.
        circuit_heads: List of (layer, head) tuples.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        method: Ablation method ("zero" or "mean").
        minimality_threshold: Threshold for minimality check.

    Returns:
        Dict with:
        - "faithfulness": How well the circuit reproduces full model behavior
        - "completeness": How much behavior is lost when circuit is ablated
        - "minimality": List of per-head redundancy results
        - "n_redundant": Number of redundant heads
        - "n_circuit_heads": Total heads in circuit
        - "baseline_metric": Full model metric value
        - "circuit_metric": Circuit-only metric value
    """
    # Baseline metrics
    full_values = []
    circuit_values = []
    for tokens in token_sequences:
        logits = model(tokens)
        full_values.append(metric_fn(logits))

        logits = _ablate_heads_outside_circuit(model, tokens, circuit_heads, method)
        circuit_values.append(metric_fn(logits))

    faith = faithfulness_score(model, circuit_heads, token_sequences, metric_fn, method)
    comp = completeness_score(model, circuit_heads, token_sequences, metric_fn, method)
    mini = minimality_check(
        model, circuit_heads, token_sequences, metric_fn, minimality_threshold, method
    )

    n_redundant = sum(1 for r in mini if r["redundant"])

    return {
        "faithfulness": faith,
        "completeness": comp,
        "minimality": mini,
        "n_redundant": n_redundant,
        "n_circuit_heads": len(circuit_heads),
        "baseline_metric": float(np.mean(full_values)),
        "circuit_metric": float(np.mean(circuit_values)),
    }
