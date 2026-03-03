"""Batch experiment utilities for systematic interpretability experiments.

Provides high-level tools for running experiments across many prompts:
- run_on_dataset: Run a metric on each prompt and aggregate
- sweep_ablations: Systematically ablate each head/layer
- find_circuit: Automatic circuit discovery via iterative ablation
- ExperimentResult: Structured result container
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.patching import ablate_heads


@dataclass
class ExperimentResult:
    """Container for experiment results with metadata.

    Attributes:
        values: The main result array or dict.
        metadata: Dict of experiment parameters and info.
        per_prompt: Optional per-prompt results before aggregation.
    """
    values: np.ndarray | dict
    metadata: dict = field(default_factory=dict)
    per_prompt: Optional[list] = None

    def summary(self) -> str:
        """Return a text summary of the result."""
        lines = []
        for key, val in self.metadata.items():
            lines.append(f"  {key}: {val}")
        if isinstance(self.values, np.ndarray):
            lines.append(f"  shape: {self.values.shape}")
            lines.append(f"  mean: {self.values.mean():.4f}")
            lines.append(f"  std: {self.values.std():.4f}")
        return "\n".join(lines)


def run_on_dataset(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    aggregate: str = "mean",
) -> ExperimentResult:
    """Run a metric function on each prompt and aggregate.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float to evaluate.
        aggregate: How to aggregate ("mean", "median", "all").

    Returns:
        ExperimentResult with aggregated values and per-prompt breakdown.
    """
    per_prompt = []
    for tokens in token_sequences:
        logits = model(tokens)
        per_prompt.append(metric_fn(logits))

    values = np.array(per_prompt)
    if aggregate == "mean":
        agg_value = np.mean(values)
    elif aggregate == "median":
        agg_value = np.median(values)
    elif aggregate == "all":
        agg_value = values
    else:
        raise ValueError(f"Unknown aggregate: {aggregate!r}")

    return ExperimentResult(
        values=np.array(agg_value),
        metadata={
            "n_prompts": len(token_sequences),
            "aggregate": aggregate,
        },
        per_prompt=per_prompt,
    )


def sweep_ablations(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    method: str = "zero",
    component: str = "heads",
) -> ExperimentResult:
    """Systematically ablate each component and measure the metric.

    Runs the ablation for each prompt and averages across prompts.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        method: Ablation method ("zero" or "mean").
        component: What to ablate ("heads" or "layers").

    Returns:
        ExperimentResult with [n_layers, n_heads] or [n_layers] ablation results.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if component == "heads":
        all_results = []
        for tokens in token_sequences:
            result = ablate_heads(model, tokens, metric_fn, method=method)
            all_results.append(result)
        avg_results = np.mean(all_results, axis=0)

        return ExperimentResult(
            values=avg_results,
            metadata={
                "method": method,
                "component": component,
                "n_prompts": len(token_sequences),
                "shape": f"[{n_layers}, {n_heads}]",
            },
            per_prompt=all_results,
        )

    elif component == "layers":
        all_results = []
        for tokens in token_sequences:
            layer_results = np.zeros(n_layers)
            _, cache = model.run_with_cache(tokens)

            for layer in range(n_layers):
                hook_name = f"blocks.{layer}.hook_attn_out"
                if method == "zero":
                    def ablate_hook(x, name):
                        return jnp.zeros_like(x)
                else:
                    attn_out = cache[hook_name]
                    mean_val = jnp.mean(attn_out, axis=0, keepdims=True) * jnp.ones_like(attn_out)
                    def ablate_hook(x, name, _m=mean_val):
                        return _m

                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate_hook)])
                layer_results[layer] = metric_fn(logits)
            all_results.append(layer_results)

        avg_results = np.mean(all_results, axis=0)
        return ExperimentResult(
            values=avg_results,
            metadata={
                "method": method,
                "component": component,
                "n_prompts": len(token_sequences),
            },
            per_prompt=all_results,
        )

    else:
        raise ValueError(f"Unknown component: {component!r}")


def find_circuit(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
    threshold: float = 0.1,
    method: str = "zero",
) -> ExperimentResult:
    """Automatic circuit discovery via iterative ablation.

    Identifies the minimal set of heads whose ablation significantly
    affects the metric. A head is "important" if ablating it changes
    the metric by more than the threshold.

    Algorithm:
    1. Compute baseline metric on all prompts.
    2. Ablate each head individually, measure metric change.
    3. Heads where |change| > threshold * |baseline| are in the circuit.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metric_fn: Function(logits) -> float.
        threshold: Minimum relative change to be considered important.
        method: Ablation method ("zero" or "mean").

    Returns:
        ExperimentResult with:
        - values: dict with "circuit_heads", "ablation_effects", "baseline"
    """
    # Baseline
    baseline_values = []
    for tokens in token_sequences:
        logits = model(tokens)
        baseline_values.append(metric_fn(logits))
    baseline = np.mean(baseline_values)

    # Ablate each head
    ablation_result = sweep_ablations(
        model, token_sequences, metric_fn, method=method, component="heads"
    )
    ablation_effects = ablation_result.values  # [n_layers, n_heads]

    # Compute relative change
    if abs(baseline) > 1e-10:
        rel_change = np.abs(ablation_effects - baseline) / abs(baseline)
    else:
        rel_change = np.abs(ablation_effects - baseline)

    # Find circuit heads
    circuit_heads = []
    n_layers, n_heads = ablation_effects.shape
    for l in range(n_layers):
        for h in range(n_heads):
            if rel_change[l, h] > threshold:
                circuit_heads.append((l, h, float(rel_change[l, h])))

    # Sort by importance
    circuit_heads.sort(key=lambda x: x[2], reverse=True)

    return ExperimentResult(
        values={
            "circuit_heads": circuit_heads,
            "ablation_effects": ablation_effects,
            "baseline": baseline,
            "relative_change": rel_change,
        },
        metadata={
            "threshold": threshold,
            "method": method,
            "n_prompts": len(token_sequences),
            "n_circuit_heads": len(circuit_heads),
        },
    )


def compare_metrics(
    model: HookedTransformer,
    token_sequences: list[jnp.ndarray],
    metrics: dict[str, Callable[[jnp.ndarray], float]],
) -> dict[str, ExperimentResult]:
    """Run multiple metrics on the same dataset and return all results.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        metrics: Dict mapping metric names to metric functions.

    Returns:
        Dict mapping metric names to ExperimentResult.
    """
    results = {}
    for name, metric_fn in metrics.items():
        results[name] = run_on_dataset(model, token_sequences, metric_fn)
    return results
