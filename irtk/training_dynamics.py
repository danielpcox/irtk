"""Training dynamics analysis.

Tools for understanding how models learn during training:
- detect_phase_transitions: Find sharp changes in a metric series
- grokking_analysis: Detect grokking (memorization -> generalization gap)
- circuit_formation_trajectory: Track a circuit metric across training checkpoints
- loss_landscape_slice: 1D slice of loss landscape along a direction
- weight_norm_trajectory: Track weight norms over training
- effective_rank_trajectory: Track effective dimensionality of activations across training
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer


def detect_phase_transitions(
    metrics: np.ndarray,
    window: int = 5,
    threshold: float = 2.0,
) -> dict:
    """Detect sharp changes (phase transitions) in a metric time series.

    Uses the ratio of local rate of change to the global mean rate of
    change. Points where this ratio exceeds the threshold are flagged.

    Args:
        metrics: [n_steps] array of metric values over training.
        window: Smoothing window for computing local derivatives.
        threshold: How many times the mean rate of change a local change
            must exceed to be flagged as a transition.

    Returns:
        Dict with:
        - "transition_indices": Array of indices where transitions occur
        - "transition_magnitudes": Rate of change at each transition
        - "smoothed_derivative": [n_steps - window] smoothed first derivative
        - "mean_derivative": Mean absolute rate of change
    """
    metrics = np.array(metrics, dtype=np.float64)
    n = len(metrics)

    if n < window + 1:
        return {
            "transition_indices": np.array([], dtype=int),
            "transition_magnitudes": np.array([]),
            "smoothed_derivative": np.array([]),
            "mean_derivative": 0.0,
        }

    # Compute smoothed derivative using moving average of differences
    diffs = np.diff(metrics)  # [n-1]

    # Smooth the derivative
    if len(diffs) < window:
        smoothed = diffs
    else:
        kernel = np.ones(window) / window
        smoothed = np.convolve(np.abs(diffs), kernel, mode="valid")

    mean_deriv = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0

    # Find transitions
    if mean_deriv < 1e-10:
        return {
            "transition_indices": np.array([], dtype=int),
            "transition_magnitudes": np.array([]),
            "smoothed_derivative": smoothed,
            "mean_derivative": mean_deriv,
        }

    transition_mask = smoothed > threshold * mean_deriv
    indices = np.where(transition_mask)[0]
    # Adjust indices for the offset from convolution
    offset = (window - 1) // 2 if len(diffs) >= window else 0
    indices = indices + offset

    return {
        "transition_indices": indices,
        "transition_magnitudes": smoothed[transition_mask - offset if len(transition_mask) > 0 else []],
        "smoothed_derivative": smoothed,
        "mean_derivative": mean_deriv,
    }


def grokking_analysis(
    train_losses: list[float],
    val_accs: list[float],
    memorization_threshold: float = 0.95,
    generalization_threshold: float = 0.95,
) -> dict:
    """Detect grokking: memorization followed by delayed generalization.

    Grokking is when the model first memorizes the training set (low loss)
    but doesn't generalize (low val accuracy), then later suddenly
    generalizes.

    Args:
        train_losses: Per-epoch training losses.
        val_accs: Per-epoch validation accuracies.
        memorization_threshold: Training loss threshold below which the
            model is considered to have memorized (relative to initial).
        generalization_threshold: Validation accuracy above which the
            model is considered to have generalized.

    Returns:
        Dict with:
        - "has_grokking": Whether grokking was detected
        - "memorization_epoch": Epoch when train loss dropped below threshold
        - "generalization_epoch": Epoch when val acc exceeded threshold
        - "grokking_gap": Number of epochs between memorization and generalization
        - "train_loss_curve": np.array of train losses
        - "val_acc_curve": np.array of val accuracies
    """
    train_losses = np.array(train_losses)
    val_accs = np.array(val_accs)

    n = min(len(train_losses), len(val_accs))
    if n == 0:
        return {
            "has_grokking": False,
            "memorization_epoch": None,
            "generalization_epoch": None,
            "grokking_gap": 0,
            "train_loss_curve": train_losses,
            "val_acc_curve": val_accs,
        }

    train_losses = train_losses[:n]
    val_accs = val_accs[:n]

    # Memorization threshold: when loss drops to memorization_threshold of initial
    initial_loss = train_losses[0] if train_losses[0] > 0 else 1.0
    mem_threshold = initial_loss * (1.0 - memorization_threshold)

    mem_epoch = None
    for i, loss in enumerate(train_losses):
        if loss < mem_threshold:
            mem_epoch = i
            break

    gen_epoch = None
    for i, acc in enumerate(val_accs):
        if acc >= generalization_threshold:
            gen_epoch = i
            break

    has_grokking = (mem_epoch is not None and gen_epoch is not None
                    and gen_epoch > mem_epoch)

    return {
        "has_grokking": has_grokking,
        "memorization_epoch": mem_epoch,
        "generalization_epoch": gen_epoch,
        "grokking_gap": (gen_epoch - mem_epoch) if has_grokking else 0,
        "train_loss_curve": train_losses,
        "val_acc_curve": val_accs,
    }


def circuit_formation_trajectory(
    checkpoints: dict[int, "HookedTransformer"],
    token_sequences: list[jnp.ndarray],
    circuit_metric_fn: Callable[["HookedTransformer", jnp.ndarray], float],
) -> dict:
    """Track a circuit-level metric across training checkpoints.

    Runs the metric function on each checkpoint to see how a circuit
    forms or changes during training.

    Args:
        checkpoints: {epoch: model} dict of training checkpoints.
        token_sequences: Test inputs to evaluate on.
        circuit_metric_fn: Function(model, tokens) -> float that measures
            some circuit property (e.g., induction score, composition score).

    Returns:
        Dict with:
        - "epochs": [n_checkpoints] sorted epoch numbers
        - "metrics": [n_checkpoints] metric values per checkpoint
        - "per_prompt_metrics": [n_checkpoints, n_prompts] per-prompt values
    """
    epochs = sorted(checkpoints.keys())
    all_metrics = []
    per_prompt = []

    for epoch in epochs:
        model = checkpoints[epoch]
        prompt_metrics = []
        for tokens in token_sequences:
            val = circuit_metric_fn(model, tokens)
            prompt_metrics.append(float(val))
        per_prompt.append(prompt_metrics)
        all_metrics.append(float(np.mean(prompt_metrics)))

    return {
        "epochs": np.array(epochs),
        "metrics": np.array(all_metrics),
        "per_prompt_metrics": np.array(per_prompt),
    }


def loss_landscape_slice(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    labels: jnp.ndarray,
    direction: Optional[np.ndarray] = None,
    alphas: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """Compute a 1D slice of the loss landscape along a direction.

    Perturbs all model weights along a random or specified direction and
    measures the loss at each perturbation magnitude.

    Args:
        model: HookedTransformer.
        tokens: [n_samples, seq_len] input tokens.
        labels: [n_samples] integer labels.
        direction: If None, uses a random direction. Otherwise, should be
            a pytree with the same structure as the model's parameters.
        alphas: Array of perturbation magnitudes. Default: linspace(-1, 1, 21).
        seed: Random seed for generating random direction.

    Returns:
        Dict with:
        - "alphas": [n_points] perturbation magnitudes
        - "losses": [n_points] loss values at each perturbation
        - "direction_norm": Frobenius norm of the direction
    """
    if alphas is None:
        alphas = np.linspace(-1.0, 1.0, 21)
    else:
        alphas = np.array(alphas)

    # Get model parameters
    params = eqx.filter(model, eqx.is_array)

    # Create direction (random or given)
    if direction is None:
        key = jax.random.PRNGKey(seed)
        flat_params, treedef = jax.tree.flatten(params)
        flat_dir = []
        for p in flat_params:
            key, subkey = jax.random.split(key)
            flat_dir.append(jax.random.normal(subkey, p.shape, dtype=p.dtype))
        direction_tree = jax.tree.unflatten(treedef, flat_dir)
    else:
        direction_tree = direction

    # Normalize direction
    dir_flat, _ = jax.tree.flatten(direction_tree)
    dir_norm = float(np.sqrt(sum(float(jnp.sum(d ** 2)) for d in dir_flat)))
    if dir_norm > 0:
        direction_tree = jax.tree.map(lambda d: d / dir_norm, direction_tree)
    dir_norm_after = 1.0  # normalized

    # Loss function
    def compute_loss(model, tokens_batch, labels_batch):
        def single_loss(tokens, label):
            logits = model(tokens)
            last_logits = logits[-1]
            log_probs = jax.nn.log_softmax(last_logits)
            return -log_probs[label]
        losses = jax.vmap(single_loss)(tokens_batch, labels_batch)
        return float(jnp.mean(losses))

    tokens_jnp = jnp.array(tokens)
    labels_jnp = jnp.array(labels)

    losses = []
    for alpha in alphas:
        # Perturb: model_params + alpha * direction
        perturbed_params = jax.tree.map(
            lambda p, d: p + alpha * d, params, direction_tree
        )
        perturbed_model = eqx.combine(perturbed_params, model)
        loss = compute_loss(perturbed_model, tokens_jnp, labels_jnp)
        losses.append(loss)

    return {
        "alphas": alphas,
        "losses": np.array(losses),
        "direction_norm": dir_norm,
    }


def weight_norm_trajectory(
    checkpoints: dict[int, "HookedTransformer"],
) -> dict:
    """Track weight norms across training checkpoints.

    Computes Frobenius norms of major weight matrices at each checkpoint.

    Args:
        checkpoints: {epoch: model} dict of training checkpoints.

    Returns:
        Dict with:
        - "epochs": [n_checkpoints] sorted epoch numbers
        - "total_norm": [n_checkpoints] total parameter norm
        - "per_component": dict mapping component name to [n_checkpoints] norms
    """
    epochs = sorted(checkpoints.keys())
    total_norms = []
    per_component: dict[str, list[float]] = {}

    for epoch in epochs:
        model = checkpoints[epoch]

        # Total norm
        params = eqx.filter(model, eqx.is_array)
        flat, _ = jax.tree.flatten(params)
        total = float(np.sqrt(sum(float(jnp.sum(p ** 2)) for p in flat)))
        total_norms.append(total)

        # Per-component norms
        component_norms = {}
        component_norms["W_E"] = float(jnp.linalg.norm(model.embed.W_E))
        component_norms["W_U"] = float(jnp.linalg.norm(model.unembed.W_U))

        for l, block in enumerate(model.blocks):
            component_norms[f"L{l}_W_Q"] = float(jnp.linalg.norm(block.attn.W_Q))
            component_norms[f"L{l}_W_K"] = float(jnp.linalg.norm(block.attn.W_K))
            component_norms[f"L{l}_W_V"] = float(jnp.linalg.norm(block.attn.W_V))
            component_norms[f"L{l}_W_O"] = float(jnp.linalg.norm(block.attn.W_O))
            component_norms[f"L{l}_W_in"] = float(jnp.linalg.norm(block.mlp.W_in))
            component_norms[f"L{l}_W_out"] = float(jnp.linalg.norm(block.mlp.W_out))

        for name, val in component_norms.items():
            if name not in per_component:
                per_component[name] = []
            per_component[name].append(val)

    return {
        "epochs": np.array(epochs),
        "total_norm": np.array(total_norms),
        "per_component": {k: np.array(v) for k, v in per_component.items()},
    }


def effective_rank_trajectory(
    checkpoints: dict[int, "HookedTransformer"],
    token_sequences: list[jnp.ndarray],
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Track effective dimensionality of activations across training checkpoints.

    At each checkpoint, collects activations and computes the participation
    ratio (effective rank) of the activation covariance matrix.

    Args:
        checkpoints: {epoch: model} dict of training checkpoints.
        token_sequences: Test inputs to evaluate on.
        hook_name: Hook point to collect activations from.
        pos: Position in sequence to analyze (-1 for last).

    Returns:
        Dict with:
        - "epochs": [n_checkpoints] sorted epoch numbers
        - "effective_rank": [n_checkpoints] participation ratio at each checkpoint
        - "top_eigenvalues": [n_checkpoints, k] top eigenvalues at each checkpoint
    """
    epochs = sorted(checkpoints.keys())
    ranks = []
    all_eigenvalues = []

    for epoch in epochs:
        model = checkpoints[epoch]
        acts = []
        for tokens in token_sequences:
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                acts.append(np.array(cache.cache_dict[hook_name][pos]))

        if not acts:
            ranks.append(0.0)
            all_eigenvalues.append(np.array([]))
            continue

        acts = np.stack(acts)  # [n_samples, d_model]

        # Center
        acts = acts - acts.mean(axis=0)

        # Covariance eigenvalues
        if acts.shape[0] < 2:
            ranks.append(0.0)
            all_eigenvalues.append(np.array([]))
            continue

        cov = acts.T @ acts / (acts.shape[0] - 1)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[::-1]  # descending
        eigenvalues = np.maximum(eigenvalues, 0)  # clip negative

        # Participation ratio
        total = np.sum(eigenvalues)
        if total > 1e-10:
            pr = total ** 2 / np.sum(eigenvalues ** 2)
        else:
            pr = 0.0

        ranks.append(float(pr))
        all_eigenvalues.append(eigenvalues[:10])  # top 10

    # Pad eigenvalues to same length
    max_len = max(len(ev) for ev in all_eigenvalues) if all_eigenvalues else 0
    padded = np.zeros((len(epochs), max_len))
    for i, ev in enumerate(all_eigenvalues):
        padded[i, :len(ev)] = ev

    return {
        "epochs": np.array(epochs),
        "effective_rank": np.array(ranks),
        "top_eigenvalues": padded,
    }
