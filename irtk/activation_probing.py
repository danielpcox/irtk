"""Rich activation probing tools.

Multi-class probing, nonlinear probes, concept localization through probing,
cross-layer probe transfer, and probe selectivity analysis.

Goes beyond simple linear probes to provide richer understanding of
what information is represented at each layer.
"""

import jax
import jax.numpy as jnp
import numpy as np


def multiclass_probe(model, tokens_list, labels_list, layer=-1, pos=-1, n_classes=None):
    """Train and evaluate a multi-class linear probe on activations.

    Args:
        model: HookedTransformer model.
        tokens_list: list of token arrays.
        labels_list: list of int labels (one per token array).
        layer: Layer to probe (-1 for last).
        pos: Position to probe (-1 for last).
        n_classes: Number of classes (inferred if None).

    Returns:
        dict with:
            accuracy: float (leave-one-out accuracy)
            class_accuracies: dict of class -> accuracy
            weight_matrix: [n_classes, d_model] probe weights
            most_discriminative_dims: list of (dim, importance)
            confusion_matrix: [n_classes, n_classes]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    target_layer = layer if layer >= 0 else n_layers - 1
    d_model = model.cfg.d_model

    labels = np.array(labels_list)
    if n_classes is None:
        n_classes = int(np.max(labels)) + 1

    # Collect activations
    activations = []
    for tokens in tokens_list:
        hook_key = f"blocks.{target_layer}.hook_resid_post"
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        act = cache_state.cache.get(hook_key)
        if act is not None:
            activations.append(np.array(act)[pos])
        else:
            activations.append(np.zeros(d_model))

    X = np.array(activations)  # [n_samples, d_model]
    n_samples = len(X)

    # Simple centroid classifier (no sklearn dependency)
    centroids = np.zeros((n_classes, d_model))
    counts = np.zeros(n_classes)
    for i in range(n_samples):
        centroids[labels[i]] += X[i]
        counts[labels[i]] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            centroids[c] /= counts[c]

    # Leave-one-out accuracy
    correct = 0
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_samples):
        # Compute centroid without this sample
        temp_centroids = centroids.copy()
        if counts[labels[i]] > 1:
            temp_centroids[labels[i]] = (centroids[labels[i]] * counts[labels[i]] - X[i]) / (counts[labels[i]] - 1)

        dists = np.array([np.linalg.norm(X[i] - temp_centroids[c]) for c in range(n_classes)])
        pred = int(np.argmin(dists))
        confusion[labels[i], pred] += 1
        if pred == labels[i]:
            correct += 1

    accuracy = correct / max(n_samples, 1)

    # Per-class accuracy
    class_acc = {}
    for c in range(n_classes):
        total = int(np.sum(confusion[c]))
        if total > 0:
            class_acc[c] = int(confusion[c, c]) / total
        else:
            class_acc[c] = 0.0

    # Most discriminative dimensions
    dim_importance = np.zeros(d_model)
    for d in range(d_model):
        between_var = np.var([centroids[c, d] for c in range(n_classes)])
        dim_importance[d] = between_var

    top_dims = np.argsort(-dim_importance)[:10]
    most_disc = [(int(d), float(dim_importance[d])) for d in top_dims]

    return {
        "accuracy": accuracy,
        "class_accuracies": class_acc,
        "weight_matrix": centroids,
        "most_discriminative_dims": most_disc,
        "confusion_matrix": confusion,
    }


def nonlinear_probe(model, tokens_list, labels_list, layer=-1, pos=-1, hidden_dim=None):
    """Train a simple nonlinear (2-layer) probe on activations.

    Uses a random projection followed by ReLU and a centroid classifier
    to capture nonlinear separability.

    Args:
        model: HookedTransformer model.
        tokens_list: list of token arrays.
        labels_list: list of int labels.
        layer: Layer to probe.
        pos: Position to probe.
        hidden_dim: Hidden dimension (default: d_model // 2).

    Returns:
        dict with:
            accuracy: float
            linear_accuracy: float (for comparison)
            nonlinearity_gain: float (improvement from nonlinear)
            projection_matrix: [hidden_dim, d_model]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    target_layer = layer if layer >= 0 else n_layers - 1
    d_model = model.cfg.d_model
    if hidden_dim is None:
        hidden_dim = max(d_model // 2, 4)

    labels = np.array(labels_list)
    n_classes = int(np.max(labels)) + 1

    # Collect activations
    activations = []
    for tokens in tokens_list:
        hook_key = f"blocks.{target_layer}.hook_resid_post"
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        act = cache_state.cache.get(hook_key)
        if act is not None:
            activations.append(np.array(act)[pos])
        else:
            activations.append(np.zeros(d_model))

    X = np.array(activations)
    n_samples = len(X)

    # Random projection + ReLU
    rng = np.random.RandomState(42)
    W_proj = rng.randn(d_model, hidden_dim).astype(np.float32) * 0.1
    X_proj = np.maximum(0, X @ W_proj)  # [n_samples, hidden_dim]

    # Centroid classifier on projected features
    centroids_nl = np.zeros((n_classes, hidden_dim))
    centroids_l = np.zeros((n_classes, d_model))
    counts = np.zeros(n_classes)

    for i in range(n_samples):
        centroids_nl[labels[i]] += X_proj[i]
        centroids_l[labels[i]] += X[i]
        counts[labels[i]] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            centroids_nl[c] /= counts[c]
            centroids_l[c] /= counts[c]

    # Nonlinear accuracy
    correct_nl = 0
    correct_l = 0
    for i in range(n_samples):
        dists_nl = [np.linalg.norm(X_proj[i] - centroids_nl[c]) for c in range(n_classes)]
        dists_l = [np.linalg.norm(X[i] - centroids_l[c]) for c in range(n_classes)]
        if int(np.argmin(dists_nl)) == labels[i]:
            correct_nl += 1
        if int(np.argmin(dists_l)) == labels[i]:
            correct_l += 1

    nl_acc = correct_nl / max(n_samples, 1)
    l_acc = correct_l / max(n_samples, 1)

    return {
        "accuracy": nl_acc,
        "linear_accuracy": l_acc,
        "nonlinearity_gain": nl_acc - l_acc,
        "projection_matrix": W_proj,
    }


def concept_localization(model, tokens_list, labels_list, pos=-1):
    """Localize a concept across layers using probing.

    Train probes at each layer to find where a concept first becomes
    linearly separable.

    Args:
        model: HookedTransformer model.
        tokens_list: list of token arrays.
        labels_list: list of int labels.
        pos: Position to probe.

    Returns:
        dict with:
            layer_accuracies: [n_layers] probe accuracy at each layer
            emergence_layer: int (first layer where accuracy exceeds chance significantly)
            peak_layer: int (layer with highest accuracy)
            accuracy_trajectory: list of float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    labels = np.array(labels_list)
    n_classes = int(np.max(labels)) + 1
    n_samples = len(tokens_list)

    accuracies = np.zeros(n_layers)

    for target_layer in range(n_layers):
        # Collect activations
        activations = []
        for tokens in tokens_list:
            hook_key = f"blocks.{target_layer}.hook_resid_post"
            cache_state = HookState(hook_fns={}, cache={})
            model(tokens, hook_state=cache_state)
            act = cache_state.cache.get(hook_key)
            if act is not None:
                activations.append(np.array(act)[pos])
            else:
                activations.append(np.zeros(d_model))

        X = np.array(activations)

        # Centroid classifier
        centroids = np.zeros((n_classes, d_model))
        counts = np.zeros(n_classes)
        for i in range(n_samples):
            centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for c in range(n_classes):
            if counts[c] > 0:
                centroids[c] /= counts[c]

        correct = 0
        for i in range(n_samples):
            dists = [np.linalg.norm(X[i] - centroids[c]) for c in range(n_classes)]
            if int(np.argmin(dists)) == labels[i]:
                correct += 1
        accuracies[target_layer] = correct / max(n_samples, 1)

    # Emergence: first layer significantly above chance
    chance = 1.0 / n_classes
    threshold = chance + 0.1
    emergence = 0
    for l in range(n_layers):
        if accuracies[l] > threshold:
            emergence = l
            break

    return {
        "layer_accuracies": accuracies,
        "emergence_layer": emergence,
        "peak_layer": int(np.argmax(accuracies)),
        "accuracy_trajectory": accuracies.tolist(),
    }


def cross_layer_probe_transfer(model, tokens_list, labels_list, train_layer, pos=-1):
    """Test how well a probe trained at one layer transfers to others.

    Args:
        model: HookedTransformer model.
        tokens_list: list of token arrays.
        labels_list: list of int labels.
        train_layer: Layer to train the probe on.
        pos: Position to probe.

    Returns:
        dict with:
            train_accuracy: float
            transfer_accuracies: [n_layers] accuracy when applied to each layer
            best_transfer_layer: int (aside from train_layer)
            representation_similarity: [n_layers] how similar each layer is to train_layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    labels = np.array(labels_list)
    n_classes = int(np.max(labels)) + 1
    n_samples = len(tokens_list)

    # Collect activations for all layers
    all_activations = {l: [] for l in range(n_layers)}
    for tokens in tokens_list:
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        for l in range(n_layers):
            act = cache_state.cache.get(f"blocks.{l}.hook_resid_post")
            if act is not None:
                all_activations[l].append(np.array(act)[pos])
            else:
                all_activations[l].append(np.zeros(d_model))

    for l in range(n_layers):
        all_activations[l] = np.array(all_activations[l])

    # Train probe on train_layer
    X_train = all_activations[train_layer]
    centroids = np.zeros((n_classes, d_model))
    counts = np.zeros(n_classes)
    for i in range(n_samples):
        centroids[labels[i]] += X_train[i]
        counts[labels[i]] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            centroids[c] /= counts[c]

    # Test on all layers
    transfer_acc = np.zeros(n_layers)
    rep_sim = np.zeros(n_layers)

    for l in range(n_layers):
        X_test = all_activations[l]
        correct = 0
        for i in range(n_samples):
            dists = [np.linalg.norm(X_test[i] - centroids[c]) for c in range(n_classes)]
            if int(np.argmin(dists)) == labels[i]:
                correct += 1
        transfer_acc[l] = correct / max(n_samples, 1)

        # Representation similarity (CKA-like)
        A = all_activations[train_layer]
        B = all_activations[l]
        if n_samples > 1:
            A_centered = A - np.mean(A, axis=0)
            B_centered = B - np.mean(B, axis=0)
            AB = np.trace(A_centered.T @ B_centered)
            AA = np.trace(A_centered.T @ A_centered)
            BB = np.trace(B_centered.T @ B_centered)
            if AA > 1e-10 and BB > 1e-10:
                rep_sim[l] = float(AB / np.sqrt(AA * BB))

    # Best transfer (excluding train layer)
    temp = transfer_acc.copy()
    temp[train_layer] = -1
    best_transfer = int(np.argmax(temp))

    return {
        "train_accuracy": float(transfer_acc[train_layer]),
        "transfer_accuracies": transfer_acc,
        "best_transfer_layer": best_transfer,
        "representation_similarity": rep_sim,
    }


def probe_selectivity(model, tokens_list, labels_list, layer=-1, pos=-1):
    """Analyze the selectivity of a probe — does it respond to only one concept?

    Measures how cleanly the probe direction separates classes and whether
    it also correlates with other features.

    Args:
        model: HookedTransformer model.
        tokens_list: list of token arrays.
        labels_list: list of int labels.
        layer: Layer to probe.
        pos: Position to probe.

    Returns:
        dict with:
            selectivity_score: float (how selective the probe direction is)
            class_separations: dict of (class_i, class_j) -> separation distance
            dimension_usage: int (how many dimensions used by the probe)
            noise_ratio: float (within-class variance / between-class variance)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    target_layer = layer if layer >= 0 else n_layers - 1
    d_model = model.cfg.d_model

    labels = np.array(labels_list)
    n_classes = int(np.max(labels)) + 1
    n_samples = len(tokens_list)

    # Collect activations
    activations = []
    for tokens in tokens_list:
        hook_key = f"blocks.{target_layer}.hook_resid_post"
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        act = cache_state.cache.get(hook_key)
        if act is not None:
            activations.append(np.array(act)[pos])
        else:
            activations.append(np.zeros(d_model))

    X = np.array(activations)

    # Compute centroids
    centroids = np.zeros((n_classes, d_model))
    counts = np.zeros(n_classes)
    for i in range(n_samples):
        centroids[labels[i]] += X[i]
        counts[labels[i]] += 1
    for c in range(n_classes):
        if counts[c] > 0:
            centroids[c] /= counts[c]

    # Between-class variance
    global_mean = np.mean(X, axis=0)
    between_var = 0.0
    for c in range(n_classes):
        if counts[c] > 0:
            between_var += counts[c] * np.sum((centroids[c] - global_mean) ** 2)
    between_var /= max(n_samples, 1)

    # Within-class variance
    within_var = 0.0
    for i in range(n_samples):
        within_var += np.sum((X[i] - centroids[labels[i]]) ** 2)
    within_var /= max(n_samples, 1)

    noise_ratio = float(within_var / (between_var + 1e-10))

    # Selectivity: inverse of noise ratio, clamped
    selectivity = float(1.0 / (1.0 + noise_ratio))

    # Class separations
    class_seps = {}
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            class_seps[(i, j)] = dist

    # Dimension usage: how many dimensions have significant between-class variance
    dim_variance = np.zeros(d_model)
    for d in range(d_model):
        dim_variance[d] = np.var([centroids[c, d] for c in range(n_classes)])
    threshold = np.max(dim_variance) * 0.1
    dim_usage = int(np.sum(dim_variance > threshold))

    return {
        "selectivity_score": selectivity,
        "class_separations": class_seps,
        "dimension_usage": dim_usage,
        "noise_ratio": noise_ratio,
    }
