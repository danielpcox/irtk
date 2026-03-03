"""Linear probes for analyzing transformer representations.

Train lightweight classifiers on cached activations to test what
information the model has learned at different layers/positions.
"""

from typing import Optional, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import numpy as np


class LinearProbe(eqx.Module):
    """A linear probe (logistic regression) for classification on activations.

    Projects d_model -> n_classes, trained with cross-entropy loss.
    """

    weight: jnp.ndarray  # [d_model, n_classes]
    bias: jnp.ndarray    # [n_classes]

    def __init__(self, d_model: int, n_classes: int, *, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (d_model, n_classes)) * 0.02
        self.bias = jnp.zeros(n_classes)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: [..., d_model] activations.

        Returns:
            [..., n_classes] logits.
        """
        return x @ self.weight + self.bias

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict class labels.

        Args:
            x: [..., d_model] activations.

        Returns:
            [...] integer class predictions.
        """
        return jnp.argmax(self(x), axis=-1)

    def accuracy(self, x: jnp.ndarray, labels: jnp.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            x: [n_samples, d_model] activations.
            labels: [n_samples] integer labels.

        Returns:
            Scalar accuracy.
        """
        return float((self.predict(x) == labels).mean())


class RegressionProbe(eqx.Module):
    """A linear regression probe for continuous-valued targets."""

    weight: jnp.ndarray  # [d_model, n_outputs]
    bias: jnp.ndarray    # [n_outputs]

    def __init__(self, d_model: int, n_outputs: int = 1, *, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (d_model, n_outputs)) * 0.02
        self.bias = jnp.zeros(n_outputs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weight + self.bias


@dataclass
class ProbeTrainResult:
    """Results from training a probe."""

    probe: eqx.Module
    train_losses: list[float]
    train_accs: list[float]
    val_losses: list[float]
    val_accs: list[float]
    best_val_acc: float
    best_epoch: int


def train_linear_probe(
    activations: jnp.ndarray,
    labels: jnp.ndarray,
    n_classes: int,
    val_frac: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 42,
    verbose: bool = True,
) -> ProbeTrainResult:
    """Train a linear probe on activations.

    Args:
        activations: [n_samples, d_model] activation vectors.
        labels: [n_samples] integer class labels.
        n_classes: Number of classes.
        val_frac: Fraction of data for validation.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        ProbeTrainResult with trained probe and metrics.
    """
    activations = jnp.array(activations)
    labels = jnp.array(labels)
    n_samples = activations.shape[0]
    d_model = activations.shape[1]

    # Train/val split
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_samples)
    n_val = max(1, int(n_samples * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train, y_train = activations[train_idx], labels[train_idx]
    x_val, y_val = activations[val_idx], labels[val_idx]

    # Create probe
    key, subkey = jax.random.split(key)
    probe = LinearProbe(d_model, n_classes, key=subkey)

    # Optimizer
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

    def loss_fn(probe, x, y):
        logits = probe(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @eqx.filter_jit
    def step(probe, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(probe, x, y)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(probe, eqx.is_array)
        )
        probe = eqx.apply_updates(probe, updates)
        return probe, opt_state_new, loss

    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    best_probe = probe
    n_train = x_train.shape[0]

    for epoch in range(epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        train_perm = jax.random.permutation(subkey, n_train)

        # Mini-batch training
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = train_perm[start:end]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            probe, opt_state, batch_loss = step(probe, opt_state, x_batch, y_batch)
            epoch_loss += float(batch_loss)
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_acc = probe.accuracy(x_train, y_train)
        val_loss = float(loss_fn(probe, x_val, y_val))
        val_acc = probe.accuracy(x_val, y_val)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_probe = probe

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    return ProbeTrainResult(
        probe=best_probe,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )
