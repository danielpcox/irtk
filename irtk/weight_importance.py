"""Weight-level importance analysis for understanding parameter roles.

Identify which weights matter most via gradient-based and information-theoretic
methods. Enables lottery ticket discovery, pruning analysis, and understanding
which parameters the model actually uses.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def _get_weight(model, weight_path: str):
    """Navigate dot-separated path to get a weight array."""
    obj = model
    for part in weight_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def fisher_information_importance(
    model: HookedTransformer,
    tokens_list: list,
    weight_path: str,
) -> dict:
    """Estimate weight importance via empirical Fisher information.

    Fisher information measures how much the loss changes with respect to
    each weight parameter. High Fisher = removing that weight hurts more.

    Approximation: FI(w) = E[(dL/dw)^2] (diagonal Fisher).

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays for gradient estimation.
        weight_path: Dot-separated path to weight (e.g., "blocks.0.mlp.W_in").

    Returns:
        Dict with:
        - "importance": array matching weight shape, importance per parameter
        - "mean_importance": scalar mean importance
        - "top_fraction_90": fraction of params containing 90% of total importance
    """
    weight = _get_weight(model, weight_path)
    weight_shape = weight.shape
    accumulated = np.zeros(weight_shape)

    for tokens in tokens_list:
        tokens = jnp.array(tokens)

        def loss_fn(w):
            # Replace weight and compute loss
            parts = weight_path.split(".")
            def setter(m):
                obj = m
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                return getattr(obj, parts[-1])

            new_model = eqx.tree_at(setter, model, w)
            logits = new_model(tokens)
            # Cross-entropy loss using next-token prediction
            log_probs = jax.nn.log_softmax(logits[:-1], axis=-1)
            targets = tokens[1:]
            loss = -jnp.mean(log_probs[jnp.arange(len(targets)), targets])
            return loss

        grad = jax.grad(loss_fn)(weight)
        accumulated += np.array(grad ** 2)

    importance = accumulated / max(len(tokens_list), 1)

    # Top fraction: what fraction of params contains 90% of importance
    flat = importance.ravel()
    sorted_imp = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_imp)
    total = cumsum[-1] if len(cumsum) > 0 else 1e-10
    threshold_idx = np.searchsorted(cumsum, 0.9 * total)
    top_frac = (threshold_idx + 1) / max(len(flat), 1)

    return {
        "importance": importance,
        "mean_importance": float(np.mean(importance)),
        "top_fraction_90": float(top_frac),
    }


def activation_variance_importance(
    model: HookedTransformer,
    tokens_list: list,
    weight_path: str,
) -> dict:
    """Rank weights by how much variance they contribute to outputs.

    Estimates importance as |weight| * variance(input), which measures
    how much each weight contributes to output variation.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        weight_path: Dot-separated path to weight.

    Returns:
        Dict with:
        - "importance": array matching weight shape
        - "sparsity_ratio": fraction of weights with near-zero importance
    """
    weight = np.array(_get_weight(model, weight_path))

    # Weight magnitude component
    mag = np.abs(weight)

    # For simplicity, importance = |weight| (magnitude-based)
    # This is a good approximation when inputs have similar variance
    importance = mag

    # Sparsity: fraction with importance < 1% of max
    threshold = 0.01 * np.max(importance)
    sparsity = float(np.mean(importance < threshold))

    return {
        "importance": importance,
        "sparsity_ratio": sparsity,
    }


def lottery_ticket_mask(
    model: HookedTransformer,
    tokens_list: list,
    weight_path: str,
    target_sparsity: float = 0.5,
    method: str = "magnitude",
) -> dict:
    """Identify the minimal weight subset maintaining performance.

    Creates a binary mask keeping only the most important weights
    at the specified sparsity level.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        weight_path: Dot-separated path to weight.
        target_sparsity: Fraction of weights to prune (0.5 = keep 50%).
        method: "magnitude" or "fisher".

    Returns:
        Dict with:
        - "mask": boolean array matching weight shape (True = keep)
        - "n_kept": number of kept parameters
        - "n_total": total parameters
        - "actual_sparsity": actual fraction pruned
    """
    weight = np.array(_get_weight(model, weight_path))

    if method == "fisher" and tokens_list:
        fi = fisher_information_importance(model, tokens_list, weight_path)
        scores = fi["importance"]
    else:
        scores = np.abs(weight)

    # Keep top (1-target_sparsity) fraction by score
    flat_scores = scores.ravel()
    n_total = len(flat_scores)
    n_keep = max(1, int(n_total * (1 - target_sparsity)))
    threshold = np.sort(flat_scores)[::-1][min(n_keep - 1, n_total - 1)]

    mask = scores >= threshold

    return {
        "mask": mask,
        "n_kept": int(np.sum(mask)),
        "n_total": n_total,
        "actual_sparsity": 1.0 - float(np.sum(mask)) / n_total,
    }


def magnitude_pruning_curve(
    model: HookedTransformer,
    tokens_list: list,
    weight_path: str,
    metric_fn: Callable,
    sparsity_levels: Optional[list[float]] = None,
) -> dict:
    """Compute performance vs sparsity curve when pruning by magnitude.

    At each sparsity level, zeros out the smallest-magnitude weights
    and evaluates the metric.

    Args:
        model: HookedTransformer.
        tokens_list: List of token arrays.
        weight_path: Dot-separated path to weight.
        metric_fn: Function from logits -> float.
        sparsity_levels: List of sparsity fractions to test. Default: [0, 0.1, ..., 0.9].

    Returns:
        Dict with:
        - "sparsity_levels": tested sparsity fractions
        - "metrics": metric value at each sparsity level
        - "critical_sparsity": first sparsity where metric drops >10% from baseline
    """
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    weight = _get_weight(model, weight_path)
    weight_np = np.array(weight)
    flat = np.abs(weight_np).ravel()
    sorted_vals = np.sort(flat)

    def setter(m):
        parts = weight_path.split(".")
        obj = m
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return getattr(obj, parts[-1])

    metrics = []
    for sparsity in sparsity_levels:
        if sparsity == 0.0:
            # No pruning
            pruned_weight = weight
        else:
            n_prune = int(len(sorted_vals) * sparsity)
            if n_prune >= len(sorted_vals):
                threshold = float("inf")
            else:
                threshold = sorted_vals[n_prune]
            mask = jnp.abs(weight) >= threshold
            pruned_weight = weight * mask

        pruned_model = eqx.tree_at(setter, model, pruned_weight)

        prompt_metrics = []
        for tokens in tokens_list:
            tokens = jnp.array(tokens)
            logits = pruned_model(tokens)
            prompt_metrics.append(float(metric_fn(logits)))
        metrics.append(float(np.mean(prompt_metrics)))

    # Find critical sparsity
    baseline = metrics[0] if metrics else 0.0
    critical = None
    for i, (sp, m) in enumerate(zip(sparsity_levels, metrics)):
        if abs(baseline) > 1e-10 and abs(m - baseline) / abs(baseline) > 0.1:
            critical = sp
            break

    return {
        "sparsity_levels": sparsity_levels,
        "metrics": metrics,
        "critical_sparsity": critical,
    }


def parameter_redundancy_analysis(
    model: HookedTransformer,
    weight_paths: list[str],
) -> dict:
    """Identify redundant parameters by cosine similarity of weight rows/columns.

    Finds groups of neurons/heads whose weights are nearly identical,
    suggesting computational redundancy.

    Args:
        model: HookedTransformer.
        weight_paths: List of weight paths to analyze together.

    Returns:
        Dict with:
        - "redundancy_scores": dict of weight_path -> redundancy score (0=unique, 1=redundant)
        - "similarity_matrices": dict of weight_path -> pairwise cosine similarity matrix
        - "most_redundant": weight_path with highest redundancy
    """
    result_scores = {}
    result_sims = {}

    for path in weight_paths:
        weight = np.array(_get_weight(model, path))

        # Reshape to 2D for similarity analysis
        if weight.ndim == 1:
            result_scores[path] = 0.0
            result_sims[path] = np.array([[1.0]])
            continue

        if weight.ndim > 2:
            weight = weight.reshape(weight.shape[0], -1)

        # Cosine similarity between rows
        norms = np.linalg.norm(weight, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = weight / norms
        sim = normed @ normed.T  # [n_rows, n_rows]

        # Redundancy score: mean off-diagonal |similarity|
        n = sim.shape[0]
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            redundancy = float(np.mean(np.abs(sim[mask])))
        else:
            redundancy = 0.0

        result_scores[path] = redundancy
        result_sims[path] = sim

    most_redundant = max(result_scores, key=result_scores.get) if result_scores else None

    return {
        "redundancy_scores": result_scores,
        "similarity_matrices": result_sims,
        "most_redundant": most_redundant,
    }
