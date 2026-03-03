"""Sparse probing methods for mechanistic interpretability.

L1-regularized probes for finding minimal feature sets that encode
concepts: sparse concept directions, feature selection, minimal
probe sets, and sparse vs dense probe comparison.

References:
- Gurnee et al. (2023) "Finding Neurons in a Haystack: Case Studies with
  Sparse Probing"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def _get_activations(model, tokens_list, layer, pos):
    """Helper to collect activations."""
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    acts = []
    for tok in tokens_list:
        cache = {}
        hs = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hs)
        acts.append(np.array(cache[hook_name][pos]))
    return np.stack(acts)


def sparse_linear_probe(
    model,
    tokens_list: list,
    labels: list,
    layer: int = -1,
    pos: int = -1,
    l1_strength: float = 0.1,
    n_steps: int = 200,
    lr: float = 0.01,
) -> dict:
    """Train an L1-regularized linear probe.

    Finds a sparse set of dimensions that predict the label.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        labels: Binary labels (0 or 1).
        layer: Layer to probe.
        pos: Position to probe.
        l1_strength: L1 regularization strength.
        n_steps: Training steps.
        lr: Learning rate.

    Returns:
        Dict with accuracy, weights, sparsity, active_dimensions.
    """
    X = _get_activations(model, tokens_list, layer, pos)
    y = np.array(labels, dtype=np.float32)
    d = X.shape[1]

    # L1-regularized logistic regression via proximal gradient
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_steps):
        # Forward
        logits = X @ w + b
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))

        # Gradient
        grad_w = X.T @ (probs - y) / len(y)
        grad_b = np.mean(probs - y)

        # Gradient step
        w = w - lr * grad_w
        b = b - lr * grad_b

        # Proximal step (soft thresholding for L1)
        w = np.sign(w) * np.maximum(np.abs(w) - lr * l1_strength, 0)

    # Evaluate
    preds = (X @ w + b > 0).astype(np.float32)
    accuracy = float(np.mean(preds == y))

    active = np.where(np.abs(w) > 1e-6)[0]
    sparsity = 1.0 - len(active) / d

    return {
        "accuracy": accuracy,
        "weights": jnp.array(w),
        "bias": float(b),
        "sparsity": sparsity,
        "active_dimensions": active.tolist(),
        "n_active": len(active),
    }


def sparse_concept_direction(
    model,
    tokens_list: list,
    labels: list,
    layer: int = -1,
    pos: int = -1,
    max_dims: int = 5,
) -> dict:
    """Find the sparsest direction that separates two concepts.

    Iteratively selects the most discriminative dimensions.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        labels: Binary labels.
        layer: Layer to analyze.
        pos: Position to analyze.
        max_dims: Maximum dimensions in the sparse direction.

    Returns:
        Dict with direction, selected_dims, accuracy_curve, separation.
    """
    X = _get_activations(model, tokens_list, layer, pos)
    y = np.array(labels, dtype=np.float32)
    d = X.shape[1]

    # Mean difference direction
    mask_0 = y == 0
    mask_1 = y == 1
    mean_0 = np.mean(X[mask_0], axis=0)
    mean_1 = np.mean(X[mask_1], axis=0)
    diff = mean_1 - mean_0

    # Select dimensions by magnitude of difference
    dim_importance = np.abs(diff)
    selected_order = np.argsort(dim_importance)[::-1]

    # Build sparse direction incrementally
    accuracy_curve = []
    for k in range(1, max_dims + 1):
        dims = selected_order[:k]
        sparse_dir = np.zeros(d)
        sparse_dir[dims] = diff[dims]
        sparse_dir /= np.linalg.norm(sparse_dir) + 1e-10

        projections = X @ sparse_dir
        threshold = np.median(projections)
        preds = (projections > threshold).astype(np.float32)
        acc = float(np.mean(preds == y))
        accuracy_curve.append(acc)

    # Final direction with max_dims
    best_dims = selected_order[:max_dims].tolist()
    final_dir = np.zeros(d)
    final_dir[best_dims] = diff[best_dims]
    final_dir /= np.linalg.norm(final_dir) + 1e-10

    # Separation
    proj_0 = np.mean(X[mask_0] @ final_dir)
    proj_1 = np.mean(X[mask_1] @ final_dir)
    separation = float(abs(proj_1 - proj_0))

    return {
        "direction": jnp.array(final_dir),
        "selected_dims": best_dims,
        "accuracy_curve": accuracy_curve,
        "separation": separation,
        "dim_importances": jnp.array(dim_importance),
    }


def feature_selection_probe(
    model,
    tokens_list: list,
    labels: list,
    layer: int = -1,
    pos: int = -1,
    method: str = "greedy",
    max_features: int = 10,
) -> dict:
    """Select minimal features for a classification task.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        labels: Labels.
        layer: Layer to probe.
        pos: Position to probe.
        method: "greedy" for forward selection.
        max_features: Max features to select.

    Returns:
        Dict with selected_features, accuracy_per_feature, final_accuracy.
    """
    X = _get_activations(model, tokens_list, layer, pos)
    y = np.array(labels, dtype=np.float32)
    d = X.shape[1]

    selected = []
    remaining = list(range(d))
    accuracy_per_step = []

    for _ in range(min(max_features, d)):
        best_feat = None
        best_acc = -1

        for feat in remaining:
            dims = selected + [feat]
            X_sub = X[:, dims]
            # Simple linear classifier
            mean_diff = np.mean(X_sub[y == 1], axis=0) - np.mean(X_sub[y == 0], axis=0)
            projections = X_sub @ mean_diff
            threshold = np.median(projections)
            preds = (projections > threshold).astype(np.float32)
            acc = float(np.mean(preds == y))
            if acc > best_acc:
                best_acc = acc
                best_feat = feat

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            accuracy_per_step.append(best_acc)

        if best_acc >= 1.0:
            break

    return {
        "selected_features": selected,
        "accuracy_per_feature": accuracy_per_step,
        "final_accuracy": accuracy_per_step[-1] if accuracy_per_step else 0.0,
        "n_features": len(selected),
    }


def minimal_probe_set(
    model,
    tokens_list: list,
    labels: list,
    layer: int = -1,
    pos: int = -1,
    target_accuracy: float = 0.8,
) -> dict:
    """Find the minimal set of dimensions needed to reach target accuracy.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        labels: Labels.
        layer: Layer to probe.
        pos: Position to probe.
        target_accuracy: Accuracy threshold.

    Returns:
        Dict with minimal_dims, accuracy_at_threshold, full_accuracy.
    """
    X = _get_activations(model, tokens_list, layer, pos)
    y = np.array(labels, dtype=np.float32)
    d = X.shape[1]

    # Rank dimensions by discriminative power
    mask_0, mask_1 = y == 0, y == 1
    mean_0, mean_1 = np.mean(X[mask_0], axis=0), np.mean(X[mask_1], axis=0)
    std_pooled = np.sqrt((np.var(X[mask_0], axis=0) + np.var(X[mask_1], axis=0)) / 2) + 1e-10
    t_scores = np.abs(mean_1 - mean_0) / std_pooled
    ranked = np.argsort(t_scores)[::-1]

    # Full accuracy
    full_dir = mean_1 - mean_0
    full_dir /= np.linalg.norm(full_dir) + 1e-10
    full_proj = X @ full_dir
    full_preds = (full_proj > np.median(full_proj)).astype(np.float32)
    full_accuracy = float(np.mean(full_preds == y))

    # Find minimal set
    minimal_dims = []
    accuracy_reached = 0.0
    for i in range(d):
        minimal_dims.append(int(ranked[i]))
        sparse_dir = np.zeros(d)
        sparse_dir[minimal_dims] = (mean_1 - mean_0)[minimal_dims]
        sparse_dir /= np.linalg.norm(sparse_dir) + 1e-10

        proj = X @ sparse_dir
        preds = (proj > np.median(proj)).astype(np.float32)
        acc = float(np.mean(preds == y))
        accuracy_reached = acc

        if acc >= target_accuracy:
            break

    return {
        "minimal_dims": minimal_dims,
        "n_dims_needed": len(minimal_dims),
        "accuracy_at_threshold": accuracy_reached,
        "full_accuracy": full_accuracy,
        "target_accuracy": target_accuracy,
    }


def sparse_vs_dense_comparison(
    model,
    tokens_list: list,
    labels: list,
    layer: int = -1,
    pos: int = -1,
    n_sparse_dims: int = 5,
) -> dict:
    """Compare sparse and dense probes.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        labels: Labels.
        layer: Layer to probe.
        pos: Position.
        n_sparse_dims: Dimensions for sparse probe.

    Returns:
        Dict with sparse_accuracy, dense_accuracy, gap,
        sparse_dims, efficiency_ratio.
    """
    X = _get_activations(model, tokens_list, layer, pos)
    y = np.array(labels, dtype=np.float32)
    d = X.shape[1]

    mask_0, mask_1 = y == 0, y == 1
    mean_0 = np.mean(X[mask_0], axis=0)
    mean_1 = np.mean(X[mask_1], axis=0)

    # Dense probe
    dense_dir = mean_1 - mean_0
    dense_dir /= np.linalg.norm(dense_dir) + 1e-10
    dense_proj = X @ dense_dir
    dense_preds = (dense_proj > np.median(dense_proj)).astype(np.float32)
    dense_acc = float(np.mean(dense_preds == y))

    # Sparse probe
    diff = mean_1 - mean_0
    top_dims = np.argsort(np.abs(diff))[::-1][:n_sparse_dims]
    sparse_dir = np.zeros(d)
    sparse_dir[top_dims] = diff[top_dims]
    sparse_dir /= np.linalg.norm(sparse_dir) + 1e-10
    sparse_proj = X @ sparse_dir
    sparse_preds = (sparse_proj > np.median(sparse_proj)).astype(np.float32)
    sparse_acc = float(np.mean(sparse_preds == y))

    gap = dense_acc - sparse_acc
    efficiency = sparse_acc / max(dense_acc, 1e-10)

    return {
        "sparse_accuracy": sparse_acc,
        "dense_accuracy": dense_acc,
        "gap": gap,
        "sparse_dims": top_dims.tolist(),
        "n_sparse_dims": n_sparse_dims,
        "n_dense_dims": d,
        "efficiency_ratio": efficiency,
    }
