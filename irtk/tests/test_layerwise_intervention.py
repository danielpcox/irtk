"""Tests for layerwise_intervention module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layerwise_intervention import (
    activation_addition,
    scaling_experiment,
    direction_intervention,
    cross_layer_transfer,
    intervention_sweep,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20])


@pytest.fixture
def metric_fn():
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


@pytest.fixture
def direction(model):
    d_model = model.cfg.d_model
    rng = np.random.RandomState(42)
    d = rng.randn(d_model).astype(np.float32)
    return d / np.linalg.norm(d)


class TestActivationAddition:
    def test_basic(self, model, tokens, direction):
        result = activation_addition(model, tokens, direction, layer=0)
        assert "original_logits" in result
        assert "modified_logits" in result
        assert "logit_change" in result
        assert "top_promoted_tokens" in result

    def test_with_metric(self, model, tokens, direction, metric_fn):
        result = activation_addition(model, tokens, direction, layer=0, metric_fn=metric_fn)
        assert "metric_change" in result

    def test_zero_scale_no_change(self, model, tokens, direction):
        result = activation_addition(model, tokens, direction, layer=0, scale=0.0)
        assert np.allclose(result["original_logits"], result["modified_logits"], atol=1e-5)


class TestScalingExperiment:
    def test_basic(self, model, tokens, metric_fn):
        result = scaling_experiment(model, tokens, layer=0, metric_fn=metric_fn)
        assert "scales" in result
        assert "metrics" in result
        assert "base_metric" in result
        assert "sensitivity" in result
        assert "optimal_scale" in result

    def test_lengths_match(self, model, tokens, metric_fn):
        result = scaling_experiment(model, tokens, layer=0, metric_fn=metric_fn)
        assert len(result["scales"]) == len(result["metrics"])

    def test_custom_scales(self, model, tokens, metric_fn):
        scales = [0.5, 1.0, 1.5]
        result = scaling_experiment(model, tokens, layer=0, metric_fn=metric_fn, scales=scales)
        assert len(result["scales"]) == 3


class TestDirectionIntervention:
    def test_basic(self, model, tokens, direction, metric_fn):
        result = direction_intervention(model, tokens, direction, metric_fn)
        assert "layer_effects" in result
        assert "best_layer" in result
        assert "best_scale" in result
        assert "effect_profile" in result

    def test_shapes(self, model, tokens, direction, metric_fn):
        result = direction_intervention(model, tokens, direction, metric_fn)
        nl = model.cfg.n_layers
        assert result["effect_profile"].shape == (nl,)


class TestCrossLayerTransfer:
    def test_basic(self, model, tokens, metric_fn):
        result = cross_layer_transfer(model, tokens, source_layer=0, target_layer=1, metric_fn=metric_fn)
        assert "base_metric" in result
        assert "transferred_metric" in result
        assert "metric_change" in result
        assert "cosine_similarity" in result

    def test_same_layer(self, model, tokens, metric_fn):
        result = cross_layer_transfer(model, tokens, source_layer=0, target_layer=0, metric_fn=metric_fn)
        # Patching same layer should have minimal effect (pre vs post may differ)
        assert "metric_change" in result

    def test_norms_positive(self, model, tokens, metric_fn):
        result = cross_layer_transfer(model, tokens, source_layer=0, target_layer=1, metric_fn=metric_fn)
        assert result["source_norm"] >= 0
        assert result["target_norm"] >= 0


class TestInterventionSweep:
    def test_basic(self, model, tokens, metric_fn):
        result = intervention_sweep(model, tokens, metric_fn, n_directions=3)
        assert "layer_sensitivity" in result
        assert "layer_max_effect" in result
        assert "most_sensitive_layer" in result
        assert "least_sensitive_layer" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = intervention_sweep(model, tokens, metric_fn, n_directions=3)
        nl = model.cfg.n_layers
        assert result["layer_sensitivity"].shape == (nl,)
        assert result["layer_max_effect"].shape == (nl,)

    def test_sensitivity_nonneg(self, model, tokens, metric_fn):
        result = intervention_sweep(model, tokens, metric_fn, n_directions=3)
        assert np.all(result["layer_sensitivity"] >= 0)
