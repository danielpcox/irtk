"""Tests for backup and redundancy detection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.backup_detection import (
    detect_backup_heads,
    knockout_compensation,
    circuit_redundancy_map,
    critical_vs_backup,
    ablation_recovery_curve,
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


def _metric(logits):
    return float(logits[-1, 0])


# ─── Detect Backup Heads ────────────────────────────────────────────────────


class TestDetectBackupHeads:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_backup_heads(model, tokens, _metric, target_layer=0, target_head=0)
        assert "backup_heads" in result
        assert "compensation_scores" in result
        assert "clean_metric" in result
        assert "ablated_metric" in result

    def test_backup_heads_are_list(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_backup_heads(model, tokens, _metric, target_layer=0, target_head=0)
        assert isinstance(result["backup_heads"], list)

    def test_compensation_scores_are_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_backup_heads(model, tokens, _metric, target_layer=0, target_head=0)
        assert isinstance(result["compensation_scores"], dict)
        # Should not include the target head itself
        assert (0, 0) not in result["compensation_scores"]


# ─── Knockout Compensation ──────────────────────────────────────────────────


class TestKnockoutCompensation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knockout_compensation(model, tokens, _metric)
        assert "per_head_ablation_effect" in result
        assert "most_compensated" in result
        assert "compensation_estimate" in result

    def test_effect_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knockout_compensation(model, tokens, _metric)
        assert result["per_head_ablation_effect"].shape == (2, 4)

    def test_compensation_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knockout_compensation(model, tokens, _metric)
        assert np.all(result["compensation_estimate"] >= 0)


# ─── Circuit Redundancy Map ────────────────────────────────────────────────


class TestCircuitRedundancyMap:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = circuit_redundancy_map(model, tokens, _metric)
        assert "redundancy_matrix" in result
        assert "head_labels" in result
        assert "most_redundant_pair" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = circuit_redundancy_map(model, tokens, _metric)
        # 2 layers * 4 heads = 8 total heads
        assert result["redundancy_matrix"].shape == (8, 8)

    def test_symmetric(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = circuit_redundancy_map(model, tokens, _metric)
        np.testing.assert_allclose(
            result["redundancy_matrix"],
            result["redundancy_matrix"].T,
            atol=1e-6,
        )


# ─── Critical vs Backup ────────────────────────────────────────────────────


class TestCriticalVsBackup:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_vs_backup(model, tokens, _metric)
        assert "critical_heads" in result
        assert "backup_heads" in result
        assert "neutral_heads" in result
        assert "classification" in result

    def test_all_heads_classified(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_vs_backup(model, tokens, _metric)
        total = (len(result["critical_heads"]) +
                 len(result["backup_heads"]) +
                 len(result["neutral_heads"]))
        assert total == 8  # 2 layers * 4 heads

    def test_classification_values(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_vs_backup(model, tokens, _metric)
        for v in result["classification"].values():
            assert v in ("critical", "backup", "neutral")


# ─── Ablation Recovery Curve ────────────────────────────────────────────────


class TestAblationRecoveryCurve:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_recovery_curve(
            model, tokens, _metric, ablate_heads=[(0, 0), (0, 1)]
        )
        assert "recovery_curve" in result
        assert "restoration_order" in result
        assert "clean_metric" in result
        assert "fully_ablated_metric" in result

    def test_curve_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        heads = [(0, 0), (0, 1), (1, 0)]
        result = ablation_recovery_curve(model, tokens, _metric, ablate_heads=heads)
        # Curve has n+1 points: fully ablated + one per restoration
        assert len(result["recovery_curve"]) == 4

    def test_restoration_order_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        heads = [(0, 0), (1, 0)]
        result = ablation_recovery_curve(model, tokens, _metric, ablate_heads=heads)
        assert len(result["restoration_order"]) == 2
