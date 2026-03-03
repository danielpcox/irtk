"""Tests for outlier dimension analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.outlier_dimensions import (
    detect_outlier_dimensions,
    outlier_magnitude_across_layers,
    outlier_removal_effect,
    attention_sink_analysis,
    dimension_utilization_spectrum,
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


class TestDetectOutlierDimensions:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_outlier_dimensions(model, tokens, layer=0)
        assert "outlier_dims" in result
        assert "magnitudes" in result
        assert "median_magnitude" in result
        assert "outlier_ratio" in result

    def test_magnitudes_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_outlier_dimensions(model, tokens, layer=0)
        assert len(result["magnitudes"]) == 16

    def test_ratio_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_outlier_dimensions(model, tokens, layer=0)
        assert 0.0 <= result["outlier_ratio"] <= 1.0


class TestOutlierMagnitudeAcrossLayers:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_magnitude_across_layers(model, tokens, target_dims=[0, 1, 2])
        assert "tracked_dims" in result
        assert "magnitude_trajectories" in result
        assert "emergence_layers" in result
        assert "growth_rates" in result

    def test_trajectory_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_magnitude_across_layers(model, tokens, target_dims=[0, 5])
        for d in [0, 5]:
            assert len(result["magnitude_trajectories"][d]) == 2  # n_layers

    def test_auto_detect(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_magnitude_across_layers(model, tokens)
        assert len(result["tracked_dims"]) > 0


class TestOutlierRemovalEffect:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_removal_effect(model, tokens, dims_to_clamp=[0, 1])
        assert "kl_divergence" in result
        assert "prediction_changed" in result
        assert "original_entropy" in result
        assert "clamped_entropy" in result

    def test_kl_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_removal_effect(model, tokens, dims_to_clamp=[0])
        assert result["kl_divergence"] >= -0.01  # approximately non-negative

    def test_entropy_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = outlier_removal_effect(model, tokens, dims_to_clamp=[0])
        assert result["original_entropy"] > 0


class TestAttentionSinkAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_sink_analysis(model, tokens)
        assert "attention_received" in result
        assert "sink_positions" in result
        assert "sink_vs_nonsink_ratio" in result

    def test_attention_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_sink_analysis(model, tokens)
        assert len(result["attention_received"]) == 4


class TestDimensionUtilizationSpectrum:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = dimension_utilization_spectrum(model, tokens)
        assert "variance_per_dim" in result
        assert "effective_dimensionality" in result
        assert "gini_coefficient" in result
        assert "utilization_entropy" in result

    def test_variance_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = dimension_utilization_spectrum(model, tokens)
        assert len(result["variance_per_dim"]) == 16

    def test_gini_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = dimension_utilization_spectrum(model, tokens)
        assert 0.0 <= result["gini_coefficient"] <= 1.0

    def test_effective_dim_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = dimension_utilization_spectrum(model, tokens)
        assert result["effective_dimensionality"] >= 1
