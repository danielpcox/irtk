"""Tests for layernorm_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layernorm_analysis import (
    gain_bias_decomposition,
    feature_amplification,
    norm_statistics,
    directional_effects,
    layernorm_jacobian,
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


class TestGainBiasDecomposition:
    def test_basic(self, model, tokens):
        result = gain_bias_decomposition(model, tokens, layer=0)
        assert "gain_contribution" in result
        assert "bias_contribution" in result
        assert "scale_factor" in result

    def test_gain_norm_positive(self, model, tokens):
        result = gain_bias_decomposition(model, tokens, layer=0)
        assert result["gain_norm"] > 0


class TestFeatureAmplification:
    def test_basic(self, model, tokens):
        result = feature_amplification(model, tokens, layer=0)
        assert "amplified_dims" in result
        assert "suppressed_dims" in result
        assert "mean_amplification" in result

    def test_counts(self, model, tokens):
        result = feature_amplification(model, tokens, layer=0)
        assert result["n_amplified"] + result["n_suppressed"] <= model.cfg.d_model


class TestNormStatistics:
    def test_basic(self, model, tokens):
        result = norm_statistics(model, tokens)
        assert "per_layer" in result
        assert "norm_growth_trend" in result

    def test_all_layers(self, model, tokens):
        result = norm_statistics(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers


class TestDirectionalEffects:
    def test_basic(self, model, tokens):
        result = directional_effects(model, tokens, layer=0)
        assert "pre_projection" in result
        assert "post_projection" in result
        assert "direction_preservation" in result

    def test_preservation_range(self, model, tokens):
        result = directional_effects(model, tokens, layer=0)
        assert -1.5 <= result["direction_preservation"] <= 1.5


class TestLayernormJacobian:
    def test_basic(self, model, tokens):
        result = layernorm_jacobian(model, tokens, layer=0)
        assert "jacobian_norm" in result
        assert "effective_rank" in result
        assert "condition_number" in result

    def test_positive_values(self, model, tokens):
        result = layernorm_jacobian(model, tokens, layer=0)
        assert result["jacobian_norm"] > 0
        assert result["effective_rank"] > 0
