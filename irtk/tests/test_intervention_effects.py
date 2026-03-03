"""Tests for intervention effects analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.intervention_effects import (
    activation_scaling_sensitivity,
    direction_addition_sweep,
    component_knockout_recovery,
    intervention_transferability,
    multi_layer_knockout,
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


class TestActivationScalingSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_scaling_sensitivity(model, tokens, _metric)
        assert "scales" in result
        assert "metrics" in result
        assert "baseline_metric" in result
        assert "sensitivity" in result
        assert "monotonic" in result

    def test_metrics_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_scaling_sensitivity(model, tokens, _metric, scales=[0.5, 1.0, 1.5])
        assert len(result["metrics"]) == 3

    def test_sensitivity_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_scaling_sensitivity(model, tokens, _metric)
        assert result["sensitivity"] >= 0


class TestDirectionAdditionSweep:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = direction_addition_sweep(model, tokens, direction, _metric)
        assert "coefficients" in result
        assert "metrics" in result
        assert "baseline_metric" in result
        assert "optimal_coefficient" in result
        assert "effect_range" in result

    def test_effect_range_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = direction_addition_sweep(model, tokens, direction, _metric)
        assert result["effect_range"] >= 0


class TestComponentKnockoutRecovery:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = component_knockout_recovery(model, tokens, _metric)
        assert "knockout_effects" in result
        assert "solo_metrics" in result
        assert "baseline_metric" in result
        assert "most_essential" in result
        assert "most_self_sufficient" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = component_knockout_recovery(model, tokens, _metric)
        assert result["knockout_effects"].shape == (2, 2)
        assert result["solo_metrics"].shape == (2, 2)


class TestInterventionTransferability:
    def test_returns_dict(self):
        model = _make_model()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([5, 6, 7, 8])
        result = intervention_transferability(model, tokens_a, tokens_b, _metric)
        assert "metric_a" in result
        assert "metric_b" in result
        assert "metric_b_patched" in result
        assert "transfer_fraction" in result
        assert "activation_distance" in result

    def test_distance_nonneg(self):
        model = _make_model()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([5, 6, 7, 8])
        result = intervention_transferability(model, tokens_a, tokens_b, _metric)
        assert result["activation_distance"] >= 0


class TestMultiLayerKnockout:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_layer_knockout(model, tokens, _metric)
        assert "n_knocked_out" in result
        assert "metrics" in result
        assert "baseline_metric" in result
        assert "half_performance_threshold" in result
        assert "graceful_degradation" in result

    def test_metrics_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_layer_knockout(model, tokens, _metric)
        assert len(result["metrics"]) == 3  # 0, 1, 2 layers knocked out

    def test_baseline_matches(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_layer_knockout(model, tokens, _metric)
        # First entry (0 knocked out) should equal baseline
        assert abs(result["metrics"][0] - result["baseline_metric"]) < 1e-4
