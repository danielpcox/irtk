"""Tests for probing dynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.probing_dynamics import (
    probe_accuracy_by_layer,
    probe_emergence_threshold,
    probe_calibration_curve,
    probe_mutual_information_matrix,
    control_task_selectivity,
)


def _make_model(seed=42):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(seed)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


# ─── Probe Accuracy by Layer ────────────────────────────────────────────────


class TestProbeAccuracyByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = probe_accuracy_by_layer(model, tokens, labels)
        assert "layer_accuracies" in result
        assert "accuracy_trajectory" in result
        assert "best_layer" in result
        assert "best_accuracy" in result

    def test_trajectory_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = probe_accuracy_by_layer(model, tokens, labels)
        assert len(result["accuracy_trajectory"]) == 2  # n_layers=2

    def test_accuracy_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = probe_accuracy_by_layer(model, tokens, labels)
        for acc in result["accuracy_trajectory"]:
            assert 0.0 <= acc <= 1.0


# ─── Probe Emergence Threshold ──────────────────────────────────────────────


class TestProbeEmergenceThreshold:
    def test_returns_dict(self):
        accs = [0.3, 0.5, 0.6, 0.8, 0.9]
        result = probe_emergence_threshold(accs, baseline_accuracy=0.5, threshold=0.7)
        assert "emergence_layer" in result
        assert "peak_layer" in result
        assert "accuracy_gain" in result

    def test_emergence_found(self):
        accs = [0.3, 0.5, 0.8, 0.9]
        result = probe_emergence_threshold(accs, threshold=0.7)
        assert result["emergence_layer"] == 2

    def test_no_emergence(self):
        accs = [0.3, 0.4, 0.5]
        result = probe_emergence_threshold(accs, threshold=0.9)
        assert result["emergence_layer"] == -1


# ─── Probe Calibration Curve ────────────────────────────────────────────────


class TestProbeCalibrationCurve:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = probe_calibration_curve(
            model, tokens, labels, "blocks.0.hook_resid_post"
        )
        assert "ece" in result
        assert "bin_confidences" in result
        assert "bin_accuracies" in result

    def test_ece_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = probe_calibration_curve(
            model, tokens, labels, "blocks.0.hook_resid_post"
        )
        assert result["ece"] >= 0


# ─── Probe MI Matrix ────────────────────────────────────────────────────────


class TestProbeMutualInformationMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        concepts = {
            "concept_a": np.array([0, 1, 0, 1]),
            "concept_b": np.array([1, 0, 1, 0]),
        }
        result = probe_mutual_information_matrix(
            model, tokens, concepts, "blocks.0.hook_resid_post"
        )
        assert "mi_matrix" in result
        assert "concept_names" in result
        assert "individual_accuracies" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        concepts = {
            "a": np.array([0, 1, 0, 1]),
            "b": np.array([1, 0, 1, 0]),
        }
        result = probe_mutual_information_matrix(
            model, tokens, concepts, "blocks.0.hook_resid_post"
        )
        assert result["mi_matrix"].shape == (2, 2)


# ─── Control Task Selectivity ───────────────────────────────────────────────


class TestControlTaskSelectivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = control_task_selectivity(
            model, tokens, labels, "blocks.0.hook_resid_post"
        )
        assert "selectivity" in result
        assert "task_accuracy" in result
        assert "control_accuracy" in result

    def test_task_accuracy_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 1])
        result = control_task_selectivity(
            model, tokens, labels, "blocks.0.hook_resid_post"
        )
        assert 0.0 <= result["task_accuracy"] <= 1.0
        assert 0.0 <= result["control_accuracy"] <= 1.0
