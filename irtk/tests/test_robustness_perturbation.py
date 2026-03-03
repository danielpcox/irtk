"""Tests for robustness and perturbation analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.robustness_perturbation import (
    weight_noise_tolerance,
    critical_parameter_identification,
    activation_noise_propagation,
    mode_connectivity_probe,
    brittle_vs_robust_circuits,
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


class TestWeightNoiseTolerance:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = weight_noise_tolerance(model, tokens, _metric)
        assert "noise_scales" in result
        assert "metric_values" in result
        assert "tolerance_threshold" in result
        assert "clean_metric" in result

    def test_metric_values_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = weight_noise_tolerance(model, tokens, _metric, noise_scales=[0.01, 0.1])
        assert len(result["metric_values"]) == 2


class TestCriticalParameterIdentification:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_parameter_identification(model, tokens, _metric)
        assert "parameter_sensitivity" in result
        assert "most_critical" in result
        assert "sensitivity_ranking" in result

    def test_has_parameters(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_parameter_identification(model, tokens, _metric)
        assert len(result["parameter_sensitivity"]) > 0

    def test_sensitivities_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_parameter_identification(model, tokens, _metric)
        assert all(v >= 0 for v in result["parameter_sensitivity"].values())


class TestActivationNoisePropagation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_noise_propagation(model, tokens)
        assert "output_perturbations" in result
        assert "amplification_factors" in result
        assert "most_amplifying_layer" in result

    def test_perturbations_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_noise_propagation(model, tokens)
        assert len(result["output_perturbations"]) == 2

    def test_perturbations_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_noise_propagation(model, tokens)
        assert np.all(result["output_perturbations"] >= 0)


class TestModeConnectivityProbe:
    def test_returns_dict(self):
        model_a = _make_model(42)
        model_b = _make_model(99)
        tokens = jnp.array([0, 1, 2, 3])
        result = mode_connectivity_probe(model_a, model_b, tokens, _metric, n_interpolations=3)
        assert "alphas" in result
        assert "metrics" in result
        assert "is_connected" in result
        assert "smoothness" in result

    def test_metrics_length(self):
        model_a = _make_model(42)
        model_b = _make_model(99)
        tokens = jnp.array([0, 1, 2, 3])
        result = mode_connectivity_probe(model_a, model_b, tokens, _metric, n_interpolations=5)
        assert len(result["metrics"]) == 5

    def test_smoothness_in_range(self):
        model_a = _make_model(42)
        model_b = _make_model(99)
        tokens = jnp.array([0, 1, 2, 3])
        result = mode_connectivity_probe(model_a, model_b, tokens, _metric)
        assert 0.0 <= result["smoothness"] <= 1.0


class TestBrittleVsRobustCircuits:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = brittle_vs_robust_circuits(model, tokens, _metric)
        assert "head_robustness" in result
        assert "brittle_heads" in result
        assert "robust_heads" in result
        assert "mean_robustness" in result

    def test_all_heads_classified(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = brittle_vs_robust_circuits(model, tokens, _metric)
        total = len(result["brittle_heads"]) + len(result["robust_heads"])
        assert total == 8  # 2 layers * 4 heads

    def test_robustness_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = brittle_vs_robust_circuits(model, tokens, _metric)
        assert 0.0 <= result["mean_robustness"] <= 1.0
