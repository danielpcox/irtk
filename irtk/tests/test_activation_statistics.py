"""Tests for activation distribution statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.activation_statistics import (
    layer_activation_moments,
    kurtosis_profile,
    normality_test_by_layer,
    activation_sparsity_pattern,
    multimodality_detection,
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


class TestLayerActivationMoments:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_activation_moments(model, tokens)
        assert "means" in result
        assert "variances" in result
        assert "skewnesses" in result
        assert "kurtoses" in result
        assert "layer_names" in result

    def test_array_lengths(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_activation_moments(model, tokens)
        assert len(result["means"]) == 2
        assert len(result["variances"]) == 2
        assert len(result["skewnesses"]) == 2
        assert len(result["kurtoses"]) == 2

    def test_variances_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_activation_moments(model, tokens)
        assert np.all(result["variances"] >= 0)


class TestKurtosisProfile:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kurtosis_profile(model, tokens)
        assert "layer_kurtosis" in result
        assert "max_kurtosis_layer" in result
        assert "kurtosis_trend" in result
        assert "per_dim_kurtosis" in result

    def test_per_dimension(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kurtosis_profile(model, tokens, per_dimension=True)
        assert result["per_dim_kurtosis"] is not None
        assert result["per_dim_kurtosis"].shape == (2, 16)

    def test_trend_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kurtosis_profile(model, tokens)
        assert result["kurtosis_trend"] in ("increasing", "decreasing", "flat")


class TestNormalityTestByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = normality_test_by_layer(model, tokens)
        assert "jb_statistics" in result
        assert "most_gaussian_layer" in result
        assert "least_gaussian_layer" in result
        assert "normality_scores" in result

    def test_jb_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = normality_test_by_layer(model, tokens)
        assert np.all(result["jb_statistics"] >= 0)

    def test_normality_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = normality_test_by_layer(model, tokens)
        assert np.all(result["normality_scores"] >= 0)
        assert np.all(result["normality_scores"] <= 1.0)


class TestActivationSparsityPattern:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_sparsity_pattern(model, tokens)
        assert "sparsity_ratios" in result
        assert "mean_magnitudes" in result
        assert "l1_norms" in result
        assert "sparsest_layer" in result
        assert "densest_layer" in result

    def test_sparsity_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_sparsity_pattern(model, tokens)
        assert np.all(result["sparsity_ratios"] >= 0)
        assert np.all(result["sparsity_ratios"] <= 1.0)

    def test_magnitudes_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_sparsity_pattern(model, tokens)
        assert np.all(result["mean_magnitudes"] >= 0)
        assert np.all(result["l1_norms"] >= 0)


class TestMultimodalityDetection:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multimodality_detection(model, tokens)
        assert "n_modes_per_layer" in result
        assert "most_multimodal_layer" in result
        assert "bimodality_coefficients" in result
        assert "layer_histograms" in result

    def test_modes_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multimodality_detection(model, tokens)
        assert np.all(result["n_modes_per_layer"] >= 1)

    def test_histograms_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multimodality_detection(model, tokens)
        assert len(result["layer_histograms"]) == 2
