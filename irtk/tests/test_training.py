"""Tests for training utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.training import (
    create_algorithmic_dataset,
    train_tiny_model,
    training_checkpoint_analysis,
)


class TestCreateAlgorithmicDataset:
    def test_modular_addition(self):
        data = create_algorithmic_dataset("modular_addition", n_samples=100, modulus=7)
        assert data["tokens"].shape == (100, 3)
        assert data["labels"].shape == (100,)
        assert np.all(data["labels"] < 7)
        assert np.all(data["labels"] >= 0)

    def test_modular_addition_correctness(self):
        data = create_algorithmic_dataset("modular_addition", n_samples=50, modulus=13)
        for i in range(50):
            a, b = data["tokens"][i, 0], data["tokens"][i, 1]
            expected = (a + b) % 13
            assert data["labels"][i] == expected

    def test_modular_subtraction(self):
        data = create_algorithmic_dataset("modular_subtraction", n_samples=50, modulus=7)
        assert data["tokens"].shape == (50, 3)
        for i in range(50):
            a, b = data["tokens"][i, 0], data["tokens"][i, 1]
            expected = (a - b) % 7
            assert data["labels"][i] == expected

    def test_copy(self):
        data = create_algorithmic_dataset("copy", n_samples=50, modulus=10, seq_len=4)
        assert data["tokens"].shape == (50, 5)  # seq_len + sep
        assert data["labels"].shape == (50,)

    def test_reverse(self):
        data = create_algorithmic_dataset("reverse", n_samples=50, modulus=10, seq_len=4)
        assert data["tokens"].shape == (50, 5)

    def test_sort(self):
        data = create_algorithmic_dataset("sort", n_samples=50, modulus=10, seq_len=4)
        assert data["tokens"].shape == (50, 5)
        # Labels should be the minimum of the input
        for i in range(50):
            assert data["labels"][i] == np.min(data["tokens"][i, :4])

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            create_algorithmic_dataset("invalid_task")


class TestTrainTinyModel:
    def test_basic_training(self):
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=4, d_head=4, n_heads=2, d_vocab=10,
        )
        data = create_algorithmic_dataset("modular_addition", n_samples=50, modulus=8)
        result = train_tiny_model(
            cfg, data["tokens"], data["labels"],
            epochs=3, batch_size=16, lr=1e-3,
        )
        assert len(result.train_losses) == 3
        assert result.final_epoch == 3

    def test_with_validation(self):
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=4, d_head=4, n_heads=2, d_vocab=10,
        )
        data = create_algorithmic_dataset("modular_addition", n_samples=80, modulus=8)
        result = train_tiny_model(
            cfg,
            data["tokens"][:60], data["labels"][:60],
            val_tokens=data["tokens"][60:], val_labels=data["labels"][60:],
            epochs=3, batch_size=16,
        )
        assert len(result.val_losses) == 3
        assert len(result.val_accs) == 3
        assert all(0.0 <= a <= 1.0 for a in result.val_accs)

    def test_checkpointing(self):
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=4, d_head=4, n_heads=2, d_vocab=10,
        )
        data = create_algorithmic_dataset("modular_addition", n_samples=50, modulus=8)
        result = train_tiny_model(
            cfg, data["tokens"], data["labels"],
            epochs=6, batch_size=16, checkpoint_every=3,
        )
        assert 3 in result.checkpoints
        assert 6 in result.checkpoints

    def test_model_runs(self):
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=4, d_head=4, n_heads=2, d_vocab=10,
        )
        data = create_algorithmic_dataset("modular_addition", n_samples=30, modulus=8)
        result = train_tiny_model(
            cfg, data["tokens"], data["labels"],
            epochs=2, batch_size=16,
        )
        # Trained model should run
        tokens = jnp.array(data["tokens"][0])
        logits = result.model(tokens)
        assert logits.shape == (3, 10)


class TestTrainingCheckpointAnalysis:
    def test_returns_epochs_and_metrics(self):
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=4, d_head=4, n_heads=2, d_vocab=10,
        )
        data = create_algorithmic_dataset("modular_addition", n_samples=30, modulus=8)
        result = train_tiny_model(
            cfg, data["tokens"], data["labels"],
            epochs=4, batch_size=16, checkpoint_every=2,
        )

        metric_fn = lambda logits: float(logits[-1].max())
        test_tokens = [jnp.array(data["tokens"][0])]
        analysis = training_checkpoint_analysis(result, test_tokens, metric_fn)
        assert len(analysis["epochs"]) == 2  # checkpoints at 2 and 4
        assert len(analysis["metrics"]) == 2
