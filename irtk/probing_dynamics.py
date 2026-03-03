"""Probing dynamics: how concepts form and sharpen across layers.

Tracks concept emergence, calibration, and redundancy through the network
using linear probes at each layer. Complements static probing (probes.py)
with dynamic, layer-sweep analysis.

Functions:
- probe_accuracy_by_layer: Train linear probes at each layer, return accuracy trajectory
- probe_emergence_threshold: Find earliest layer where a concept becomes decodable
- probe_calibration_curve: Expected calibration error at a given layer
- probe_mutual_information_matrix: Pairwise MI between concept probes across layers
- control_task_selectivity: Hewitt & Liang selectivity (task vs control)

References:
    - Alain & Bengio (2016) "Understanding intermediate representations"
    - Hewitt & Liang (2019) "Designing and Interpreting Probes with Control Tasks"
    - Dalvi et al. (2019) "What is one grain of sand in the desert?"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def _train_simple_probe(X, y, n_classes=2, seed=0):
    """Train a simple linear probe using least squares.

    For binary: logistic-style via thresholding.
    For multi-class: one-vs-rest with argmax.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    n_samples = X.shape[0]

    if n_classes <= 2:
        # Binary: solve via least squares
        X_aug = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        # Pseudoinverse solution
        w, _, _, _ = np.linalg.lstsq(X_aug, y.astype(np.float64), rcond=None)
        preds = (X_aug @ w > 0.5).astype(int)
        probs = 1.0 / (1.0 + np.exp(-(X_aug @ w)))
        acc = float(np.mean(preds == y))
        return acc, probs, w
    else:
        # Multi-class: one-hot + lstsq
        Y_onehot = np.eye(n_classes)[y]
        X_aug = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        W, _, _, _ = np.linalg.lstsq(X_aug, Y_onehot, rcond=None)
        logits = X_aug @ W
        preds = np.argmax(logits, axis=1)
        # Softmax for calibration
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        acc = float(np.mean(preds == y))
        return acc, probs, W


def probe_accuracy_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    labels: jnp.ndarray,
    hook_names: Optional[list] = None,
    pos: int = -1,
) -> dict:
    """Train a linear probe at each layer and return accuracy trajectory.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens (or list of token sequences).
        labels: [n_samples] integer labels for classification.
        hook_names: List of hook point names to probe. If None, uses
            residual stream post-hooks for all layers.
        pos: Token position to extract representations from (-1 = last).

    Returns:
        Dict with:
            "layer_accuracies": dict mapping hook_name -> accuracy
            "accuracy_trajectory": list of accuracies in order
            "best_layer": hook name with highest accuracy
            "best_accuracy": highest accuracy achieved
    """
    _, cache = model.run_with_cache(tokens)

    if hook_names is None:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    labels = np.array(labels)
    accuracies = {}
    trajectory = []

    for name in hook_names:
        if name not in cache.cache_dict:
            accuracies[name] = 0.0
            trajectory.append(0.0)
            continue

        act = cache.cache_dict[name]  # [seq_len, d_model]
        if act.ndim == 2:
            # Single sequence: use position
            X = np.array(act[pos:pos+1]) if pos >= 0 else np.array(act[pos:])
        else:
            X = np.array(act)

        # If we only have one sample and multiple labels, tile
        if X.shape[0] < len(labels):
            X = np.tile(X, (len(labels), 1))[:len(labels)]

        n_classes = int(np.max(labels)) + 1
        acc, _, _ = _train_simple_probe(X[:len(labels)], labels, n_classes)
        accuracies[name] = acc
        trajectory.append(acc)

    best_name = max(accuracies, key=accuracies.get) if accuracies else ""
    best_acc = max(trajectory) if trajectory else 0.0

    return {
        "layer_accuracies": accuracies,
        "accuracy_trajectory": trajectory,
        "best_layer": best_name,
        "best_accuracy": best_acc,
    }


def probe_emergence_threshold(
    layer_accuracies: list,
    baseline_accuracy: float = 0.5,
    threshold: float = 0.7,
) -> dict:
    """Find the earliest layer where probe accuracy exceeds a threshold.

    Args:
        layer_accuracies: List of accuracies per layer (in order).
        baseline_accuracy: Chance-level accuracy (e.g., 0.5 for binary).
        threshold: Accuracy threshold for emergence.

    Returns:
        Dict with:
            "emergence_layer": first layer exceeding threshold (-1 if never)
            "above_baseline_layer": first layer above baseline
            "peak_layer": layer with maximum accuracy
            "accuracy_gain": max accuracy minus baseline
    """
    layer_accuracies = list(layer_accuracies)
    emergence = -1
    above_baseline = -1

    for i, acc in enumerate(layer_accuracies):
        if acc >= threshold and emergence == -1:
            emergence = i
        if acc > baseline_accuracy + 0.01 and above_baseline == -1:
            above_baseline = i

    peak = int(np.argmax(layer_accuracies)) if layer_accuracies else -1
    gain = max(layer_accuracies) - baseline_accuracy if layer_accuracies else 0.0

    return {
        "emergence_layer": emergence,
        "above_baseline_layer": above_baseline,
        "peak_layer": peak,
        "accuracy_gain": gain,
    }


def probe_calibration_curve(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    labels: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
    n_bins: int = 10,
) -> dict:
    """Compute expected calibration error (ECE) for a probe at a layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        labels: [n_samples] integer labels.
        hook_name: Hook point to probe.
        pos: Token position (-1 = last).
        n_bins: Number of calibration bins.

    Returns:
        Dict with:
            "ece": expected calibration error (0 = perfect calibration)
            "bin_confidences": [n_bins] mean confidence per bin
            "bin_accuracies": [n_bins] actual accuracy per bin
            "bin_counts": [n_bins] number of samples per bin
    """
    _, cache = model.run_with_cache(tokens)
    labels = np.array(labels)

    if hook_name not in cache.cache_dict:
        return {
            "ece": 1.0,
            "bin_confidences": np.zeros(n_bins),
            "bin_accuracies": np.zeros(n_bins),
            "bin_counts": np.zeros(n_bins, dtype=int),
        }

    act = cache.cache_dict[hook_name]
    X = np.array(act[pos:pos+1]) if pos >= 0 else np.array(act[pos:])
    if X.shape[0] < len(labels):
        X = np.tile(X, (len(labels), 1))[:len(labels)]

    n_classes = int(np.max(labels)) + 1
    _, probs, _ = _train_simple_probe(X[:len(labels)], labels, n_classes)

    # Max confidence and correctness
    if probs.ndim == 1:
        confidences = np.abs(probs - 0.5) * 2  # Scale to [0, 1]
        preds = (probs > 0.5).astype(int)
    else:
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)

    correct = (preds == labels[:len(preds)]).astype(float)

    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (confidences >= bin_edges[b]) & (confidences < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = mask | (confidences == bin_edges[b + 1])
        if np.sum(mask) > 0:
            bin_confs[b] = np.mean(confidences[mask])
            bin_accs[b] = np.mean(correct[mask])
            bin_counts[b] = int(np.sum(mask))

    # ECE = weighted average of |confidence - accuracy|
    total = np.sum(bin_counts)
    if total > 0:
        ece = float(np.sum(bin_counts * np.abs(bin_confs - bin_accs)) / total)
    else:
        ece = 0.0

    return {
        "ece": ece,
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


def probe_mutual_information_matrix(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    concept_labels: dict,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Estimate pairwise mutual information between probe predictions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        concept_labels: {concept_name: labels_array} mapping concept names
            to their label arrays.
        hook_name: Hook point to probe.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "mi_matrix": [n_concepts, n_concepts] pairwise MI estimates
            "concept_names": list of concept names
            "individual_accuracies": {concept_name: accuracy}
    """
    _, cache = model.run_with_cache(tokens)
    concepts = list(concept_labels.keys())
    n_concepts = len(concepts)

    mi_matrix = np.zeros((n_concepts, n_concepts))
    individual_accs = {}

    if hook_name not in cache.cache_dict or n_concepts == 0:
        return {
            "mi_matrix": mi_matrix,
            "concept_names": concepts,
            "individual_accuracies": individual_accs,
        }

    act = cache.cache_dict[hook_name]
    X = np.array(act[pos:pos+1]) if pos >= 0 else np.array(act[pos:])

    # Train probes for each concept
    concept_preds = {}
    for name, labels in concept_labels.items():
        labels = np.array(labels)
        X_tiled = np.tile(X, (len(labels), 1))[:len(labels)]
        n_classes = int(np.max(labels)) + 1
        acc, probs, _ = _train_simple_probe(X_tiled, labels, n_classes)
        individual_accs[name] = acc
        if probs.ndim == 1:
            concept_preds[name] = (probs > 0.5).astype(int)
        else:
            concept_preds[name] = np.argmax(probs, axis=1)

    # Compute pairwise MI via contingency tables
    for i in range(n_concepts):
        for j in range(n_concepts):
            pred_i = concept_preds[concepts[i]]
            pred_j = concept_preds[concepts[j]]
            n = min(len(pred_i), len(pred_j))
            pred_i, pred_j = pred_i[:n], pred_j[:n]

            # Discrete MI estimate
            vals_i = np.unique(pred_i)
            vals_j = np.unique(pred_j)
            mi = 0.0
            for vi in vals_i:
                for vj in vals_j:
                    pij = np.mean((pred_i == vi) & (pred_j == vj))
                    pi = np.mean(pred_i == vi)
                    pj = np.mean(pred_j == vj)
                    if pij > 0 and pi > 0 and pj > 0:
                        mi += pij * np.log(pij / (pi * pj))
            mi_matrix[i, j] = mi

    return {
        "mi_matrix": mi_matrix,
        "concept_names": concepts,
        "individual_accuracies": individual_accs,
    }


def control_task_selectivity(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    labels: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
    n_controls: int = 5,
    seed: int = 42,
) -> dict:
    """Compute Hewitt & Liang (2019) selectivity score.

    Selectivity = task_accuracy - control_accuracy, where control uses
    random labels. High selectivity means the probe captures genuine
    structure, not just dataset statistics.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        labels: [n_samples] true labels.
        hook_name: Hook point to probe.
        pos: Token position (-1 = last).
        n_controls: Number of random control tasks to average.
        seed: Random seed for control labels.

    Returns:
        Dict with:
            "selectivity": task_accuracy - mean_control_accuracy
            "task_accuracy": accuracy on real task
            "control_accuracy": mean accuracy on random labels
            "control_std": std of control accuracies
    """
    _, cache = model.run_with_cache(tokens)
    labels = np.array(labels)

    if hook_name not in cache.cache_dict:
        return {
            "selectivity": 0.0,
            "task_accuracy": 0.0,
            "control_accuracy": 0.0,
            "control_std": 0.0,
        }

    act = cache.cache_dict[hook_name]
    X = np.array(act[pos:pos+1]) if pos >= 0 else np.array(act[pos:])
    if X.shape[0] < len(labels):
        X = np.tile(X, (len(labels), 1))[:len(labels)]

    n_classes = int(np.max(labels)) + 1
    task_acc, _, _ = _train_simple_probe(X[:len(labels)], labels, n_classes)

    # Control tasks with random labels
    rng = np.random.RandomState(seed)
    control_accs = []
    for _ in range(n_controls):
        random_labels = rng.randint(0, n_classes, size=len(labels))
        ctrl_acc, _, _ = _train_simple_probe(X[:len(labels)], random_labels, n_classes)
        control_accs.append(ctrl_acc)

    mean_ctrl = float(np.mean(control_accs))
    std_ctrl = float(np.std(control_accs))

    return {
        "selectivity": task_acc - mean_ctrl,
        "task_accuracy": task_acc,
        "control_accuracy": mean_ctrl,
        "control_std": std_ctrl,
    }
