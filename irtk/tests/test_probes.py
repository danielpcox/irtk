"""Tests for linear probes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.probes import LinearProbe, train_linear_probe


class TestLinearProbe:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        probe = LinearProbe(16, 3, key=key)
        x = jnp.ones((5, 16))
        logits = probe(x)
        assert logits.shape == (5, 3)

    def test_predict(self):
        key = jax.random.PRNGKey(0)
        probe = LinearProbe(16, 3, key=key)
        x = jnp.ones((5, 16))
        preds = probe.predict(x)
        assert preds.shape == (5,)
        assert preds.dtype in (jnp.int32, jnp.int64)

    def test_accuracy(self):
        key = jax.random.PRNGKey(0)
        probe = LinearProbe(4, 2, key=key)
        x = jnp.ones((10, 4))
        labels = jnp.zeros(10, dtype=jnp.int32)
        acc = probe.accuracy(x, labels)
        assert 0.0 <= acc <= 1.0


class TestTrainLinearProbe:
    def test_linearly_separable(self):
        """Probe should achieve high accuracy on linearly separable data."""
        key = jax.random.PRNGKey(42)
        n = 200
        d = 8

        # Create linearly separable data
        k1, k2 = jax.random.split(key)
        x_class0 = jax.random.normal(k1, (n, d)) + 2.0
        x_class1 = jax.random.normal(k2, (n, d)) - 2.0
        x = jnp.concatenate([x_class0, x_class1])
        y = jnp.concatenate([jnp.zeros(n, dtype=jnp.int32), jnp.ones(n, dtype=jnp.int32)])

        result = train_linear_probe(x, y, n_classes=2, epochs=50, verbose=False)
        assert result.best_val_acc > 0.9

    def test_returns_result_object(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 4))
        y = jnp.zeros(50, dtype=jnp.int32)
        result = train_linear_probe(x, y, n_classes=2, epochs=5, verbose=False)
        assert len(result.train_losses) == 5
        assert len(result.val_accs) == 5
        assert result.probe is not None
