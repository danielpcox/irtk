"""Tests for training dynamics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.training_dynamics import (
    detect_phase_transitions,
    grokking_analysis,
    circuit_formation_trajectory,
    loss_landscape_slice,
    weight_norm_trajectory,
    effective_rank_trajectory,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _make_checkpoints():
    """Create fake training checkpoints."""
    return {
        10: _make_model(),
        20: _make_model(),
        30: _make_model(),
    }


class TestDetectPhaseTransitions:
    def test_returns_dict(self):
        metrics = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 0.5, 0.3, 0.2, 0.15, 0.1])
        result = detect_phase_transitions(metrics, window=3)
        assert "transition_indices" in result
        assert "smoothed_derivative" in result
        assert "mean_derivative" in result

    def test_detects_sharp_drop(self):
        # Flat, then sharp drop, then flat
        metrics = np.concatenate([
            np.ones(10) * 5.0,
            np.linspace(5.0, 0.1, 3),
            np.ones(10) * 0.1,
        ])
        result = detect_phase_transitions(metrics, window=3, threshold=2.0)
        # Should detect transitions around the sharp drop
        assert len(result["transition_indices"]) > 0

    def test_no_transitions_for_constant(self):
        metrics = np.ones(20)
        result = detect_phase_transitions(metrics, window=3)
        assert len(result["transition_indices"]) == 0

    def test_short_sequence(self):
        metrics = np.array([1.0, 0.5])
        result = detect_phase_transitions(metrics, window=5)
        assert isinstance(result["transition_indices"], np.ndarray)

    def test_mean_derivative_nonnegative(self):
        metrics = np.random.randn(30)
        result = detect_phase_transitions(metrics, window=3)
        assert result["mean_derivative"] >= 0


class TestGrokkingAnalysis:
    def test_returns_dict(self):
        train_losses = [5.0, 4.0, 3.0, 1.0, 0.5, 0.1, 0.01, 0.01, 0.01, 0.01]
        val_accs = [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
        result = grokking_analysis(train_losses, val_accs)
        assert "has_grokking" in result
        assert "memorization_epoch" in result
        assert "generalization_epoch" in result
        assert "grokking_gap" in result

    def test_detects_grokking(self):
        # Train loss drops early, val acc rises late
        train_losses = [5.0] + [0.01] * 19
        val_accs = [0.1] * 15 + [0.5, 0.7, 0.9, 0.96, 0.99]
        result = grokking_analysis(train_losses, val_accs)
        assert result["has_grokking"] is True
        assert result["memorization_epoch"] < result["generalization_epoch"]
        assert result["grokking_gap"] > 0

    def test_no_grokking_simultaneous(self):
        # Both improve together
        train_losses = [5.0, 3.0, 1.0, 0.1, 0.01]
        val_accs = [0.1, 0.5, 0.8, 0.96, 0.99]
        result = grokking_analysis(train_losses, val_accs)
        # memorization and generalization happen close together
        if result["has_grokking"]:
            assert result["grokking_gap"] <= 2

    def test_empty_inputs(self):
        result = grokking_analysis([], [])
        assert result["has_grokking"] is False
        assert result["memorization_epoch"] is None
        assert result["generalization_epoch"] is None

    def test_no_generalization(self):
        train_losses = [5.0] + [0.01] * 9
        val_accs = [0.1] * 10
        result = grokking_analysis(train_losses, val_accs)
        assert result["has_grokking"] is False


class TestCircuitFormationTrajectory:
    def test_returns_dict(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]

        def metric(model, tokens):
            logits = model(tokens)
            return float(logits[-1, 0])

        result = circuit_formation_trajectory(checkpoints, tokens_list, metric)
        assert "epochs" in result
        assert "metrics" in result
        assert "per_prompt_metrics" in result

    def test_epochs_sorted(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3])]

        def metric(model, tokens):
            return 1.0

        result = circuit_formation_trajectory(checkpoints, tokens_list, metric)
        np.testing.assert_array_equal(result["epochs"], [10, 20, 30])

    def test_correct_shape(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]

        def metric(model, tokens):
            return 1.0

        result = circuit_formation_trajectory(checkpoints, tokens_list, metric)
        assert result["metrics"].shape == (3,)
        assert result["per_prompt_metrics"].shape == (3, 2)


class TestLossLandscapeSlice:
    def test_returns_dict(self):
        model = _make_model()
        tokens = np.array([[0, 1, 2], [3, 4, 5]])
        labels = np.array([10, 20])
        result = loss_landscape_slice(model, tokens, labels)
        assert "alphas" in result
        assert "losses" in result
        assert "direction_norm" in result

    def test_losses_shape(self):
        model = _make_model()
        tokens = np.array([[0, 1, 2], [3, 4, 5]])
        labels = np.array([10, 20])
        alphas = np.linspace(-0.5, 0.5, 5)
        result = loss_landscape_slice(model, tokens, labels, alphas=alphas)
        assert result["losses"].shape == (5,)
        assert result["alphas"].shape == (5,)

    def test_zero_alpha_is_baseline(self):
        model = _make_model()
        tokens = np.array([[0, 1, 2], [3, 4, 5]])
        labels = np.array([10, 20])
        # Use only alpha=0
        result = loss_landscape_slice(model, tokens, labels, alphas=np.array([0.0]))
        assert len(result["losses"]) == 1
        assert result["losses"][0] > 0  # cross-entropy should be positive

    def test_direction_norm_positive(self):
        model = _make_model()
        tokens = np.array([[0, 1, 2]])
        labels = np.array([5])
        result = loss_landscape_slice(model, tokens, labels)
        assert result["direction_norm"] > 0


class TestWeightNormTrajectory:
    def test_returns_dict(self):
        checkpoints = _make_checkpoints()
        result = weight_norm_trajectory(checkpoints)
        assert "epochs" in result
        assert "total_norm" in result
        assert "per_component" in result

    def test_epochs_sorted(self):
        checkpoints = _make_checkpoints()
        result = weight_norm_trajectory(checkpoints)
        np.testing.assert_array_equal(result["epochs"], [10, 20, 30])

    def test_total_norm_positive(self):
        checkpoints = _make_checkpoints()
        result = weight_norm_trajectory(checkpoints)
        assert all(n > 0 for n in result["total_norm"])

    def test_per_component_keys(self):
        checkpoints = _make_checkpoints()
        result = weight_norm_trajectory(checkpoints)
        assert "W_E" in result["per_component"]
        assert "W_U" in result["per_component"]
        assert "L0_W_Q" in result["per_component"]
        assert "L1_W_out" in result["per_component"]

    def test_per_component_shapes(self):
        checkpoints = _make_checkpoints()
        result = weight_norm_trajectory(checkpoints)
        for name, norms in result["per_component"].items():
            assert norms.shape == (3,)


class TestEffectiveRankTrajectory:
    def test_returns_dict(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = effective_rank_trajectory(
            checkpoints, tokens_list, "blocks.0.hook_resid_post"
        )
        assert "epochs" in result
        assert "effective_rank" in result
        assert "top_eigenvalues" in result

    def test_effective_rank_positive(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                       jnp.array([8, 9, 10, 11])]
        result = effective_rank_trajectory(
            checkpoints, tokens_list, "blocks.0.hook_resid_post"
        )
        for rank in result["effective_rank"]:
            assert rank >= 0

    def test_epochs_match(self):
        checkpoints = _make_checkpoints()
        tokens_list = [jnp.array([0, 1, 2, 3])]
        result = effective_rank_trajectory(
            checkpoints, tokens_list, "blocks.0.hook_resid_post"
        )
        np.testing.assert_array_equal(result["epochs"], [10, 20, 30])
