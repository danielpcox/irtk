"""Training utilities for mechanistic interpretability experiments.

Tools for training small models from scratch on controlled tasks:
- train_tiny_model: Train a small transformer for algorithmic tasks
- create_algorithmic_dataset: Generate data for modular arithmetic, copying, etc.
- training_checkpoint_analysis: Compare model at different training steps
"""

from typing import Optional, Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer


def create_algorithmic_dataset(
    task: str,
    n_samples: int = 1000,
    modulus: int = 113,
    seq_len: int = 3,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create training data for algorithmic tasks.

    Args:
        task: One of:
            - "modular_addition": a + b mod p -> c (classic grokking task)
            - "modular_subtraction": a - b mod p -> c
            - "copy": repeat input sequence
            - "reverse": reverse input sequence
            - "sort": sort input sequence
        n_samples: Number of samples to generate.
        modulus: Modulus for modular arithmetic tasks.
        seq_len: Sequence length for sequence tasks.
        seed: Random seed.

    Returns:
        Dict with "tokens" [n_samples, seq_len] and "labels" [n_samples].
    """
    rng = np.random.RandomState(seed)

    if task == "modular_addition":
        a = rng.randint(0, modulus, n_samples)
        b = rng.randint(0, modulus, n_samples)
        c = (a + b) % modulus
        # Token format: [a, b, =] where = is modulus
        tokens = np.stack([a, b, np.full(n_samples, modulus)], axis=1)
        return {"tokens": tokens, "labels": c}

    elif task == "modular_subtraction":
        a = rng.randint(0, modulus, n_samples)
        b = rng.randint(0, modulus, n_samples)
        c = (a - b) % modulus
        tokens = np.stack([a, b, np.full(n_samples, modulus)], axis=1)
        return {"tokens": tokens, "labels": c}

    elif task == "copy":
        data = rng.randint(0, modulus, (n_samples, seq_len))
        # Input: [seq..., sep], label: first token (simplified)
        sep = modulus
        tokens = np.concatenate([data, np.full((n_samples, 1), sep)], axis=1)
        return {"tokens": tokens, "labels": data[:, 0]}

    elif task == "reverse":
        data = rng.randint(0, modulus, (n_samples, seq_len))
        sep = modulus
        tokens = np.concatenate([data, np.full((n_samples, 1), sep)], axis=1)
        # Label: last token of input (first of reversed)
        return {"tokens": tokens, "labels": data[:, -1]}

    elif task == "sort":
        data = rng.randint(0, modulus, (n_samples, seq_len))
        sep = modulus
        tokens = np.concatenate([data, np.full((n_samples, 1), sep)], axis=1)
        # Label: minimum value
        return {"tokens": tokens, "labels": np.min(data, axis=1)}

    else:
        raise ValueError(f"Unknown task: {task!r}. Choose from: "
                        "'modular_addition', 'modular_subtraction', 'copy', 'reverse', 'sort'")


@dataclass
class TrainingResult:
    """Result of training a tiny model."""
    model: HookedTransformer
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    checkpoints: dict[int, HookedTransformer] = field(default_factory=dict)
    final_epoch: int = 0


def train_tiny_model(
    cfg: HookedTransformerConfig,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    val_tokens: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    checkpoint_every: int = 0,
    seed: int = 0,
) -> TrainingResult:
    """Train a small transformer from scratch on a supervised task.

    The model predicts the label from the logits at the last position.

    Args:
        cfg: Model configuration.
        train_tokens: [n_train, seq_len] training input tokens.
        train_labels: [n_train] integer labels.
        val_tokens: Optional validation tokens.
        val_labels: Optional validation labels.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        checkpoint_every: If > 0, save model every N epochs.
        seed: Random seed.

    Returns:
        TrainingResult with trained model and training history.
    """
    model = HookedTransformer(cfg)
    # Initialize with random weights
    key = jax.random.PRNGKey(seed)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.02)
        else:
            new_leaves.append(leaf)
    model = jax.tree.unflatten(treedef, new_leaves)

    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_train = len(train_tokens)
    result = TrainingResult(model=model)

    def loss_fn(model, tokens_batch, labels_batch):
        """Cross-entropy loss at the last position."""
        # vmap over batch
        def single_loss(tokens, label):
            logits = model(tokens)
            last_logits = logits[-1]  # [d_vocab]
            log_probs = jax.nn.log_softmax(last_logits)
            return -log_probs[label]

        losses = jax.vmap(single_loss)(tokens_batch, labels_batch)
        return jnp.mean(losses)

    @eqx.filter_jit
    def train_step(model, opt_state, tokens_batch, labels_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tokens_batch, labels_batch)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    for epoch in range(epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)
        shuffled_tokens = jnp.array(train_tokens)[perm]
        shuffled_labels = jnp.array(train_labels)[perm]

        epoch_losses = []
        for i in range(0, n_train, batch_size):
            batch_tokens = shuffled_tokens[i:i + batch_size]
            batch_labels = shuffled_labels[i:i + batch_size]
            if len(batch_tokens) == 0:
                continue
            model, opt_state, loss = train_step(model, opt_state, batch_tokens, batch_labels)
            epoch_losses.append(float(loss))

        result.train_losses.append(float(np.mean(epoch_losses)))

        # Validation
        if val_tokens is not None and val_labels is not None:
            val_loss = float(loss_fn(model, jnp.array(val_tokens), jnp.array(val_labels)))
            result.val_losses.append(val_loss)

            # Accuracy
            def predict(tokens):
                logits = model(tokens)
                return jnp.argmax(logits[-1])

            preds = jax.vmap(predict)(jnp.array(val_tokens))
            acc = float(jnp.mean(preds == jnp.array(val_labels)))
            result.val_accs.append(acc)

        # Checkpoint
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            result.checkpoints[epoch + 1] = model

    result.model = model
    result.final_epoch = epochs
    return result


def training_checkpoint_analysis(
    result: TrainingResult,
    token_sequences: list[jnp.ndarray],
    metric_fn: Callable[[jnp.ndarray], float],
) -> dict[str, np.ndarray]:
    """Analyze how a metric changes across training checkpoints.

    Args:
        result: TrainingResult with checkpoints.
        token_sequences: List of test inputs.
        metric_fn: Function(logits) -> float to evaluate.

    Returns:
        Dict with:
        - "epochs": array of checkpoint epochs
        - "metrics": array of metric values per checkpoint
    """
    epochs = sorted(result.checkpoints.keys())
    metrics = []

    for epoch in epochs:
        checkpoint = result.checkpoints[epoch]
        epoch_metrics = []
        for tokens in token_sequences:
            logits = checkpoint(tokens)
            epoch_metrics.append(metric_fn(logits))
        metrics.append(float(np.mean(epoch_metrics)))

    return {
        "epochs": np.array(epochs),
        "metrics": np.array(metrics),
    }
