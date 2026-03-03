"""Tests for superposition analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.superposition import (
    compute_feature_directions,
    feature_interference,
    dimensionality_analysis,
    activation_covariance,
    feature_sparsity,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestComputeFeatureDirections:
    def test_pca_shape(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        dirs = compute_feature_directions(activations, n_features=5, method="pca")
        assert dirs.shape == (5, 16)

    def test_pca_unit_vectors(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        dirs = compute_feature_directions(activations, n_features=5, method="pca")
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_pca_orthogonal(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        dirs = compute_feature_directions(activations, n_features=5, method="pca")
        # PCA directions should be orthogonal
        gram = dirs @ dirs.T
        expected = np.eye(5)
        np.testing.assert_allclose(gram, expected, atol=1e-5)

    def test_random_shape(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(50, 8)
        dirs = compute_feature_directions(activations, n_features=10, method="random")
        assert dirs.shape == (10, 8)

    def test_random_unit_vectors(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(50, 8)
        dirs = compute_feature_directions(activations, n_features=10, method="random")
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_default_n_features(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(50, 8)
        dirs = compute_feature_directions(activations, method="pca")
        assert dirs.shape == (8, 8)

    def test_invalid_method(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(50, 8)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_feature_directions(activations, method="invalid")


class TestFeatureInterference:
    def test_shape(self):
        dirs = np.eye(5, 8)  # 5 orthogonal directions in 8-D
        sim = feature_interference(dirs)
        assert sim.shape == (5, 5)

    def test_orthogonal_no_interference(self):
        dirs = np.eye(5, 8)
        sim = feature_interference(dirs)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)
        off_diag = sim - np.diag(np.diag(sim))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-5)

    def test_identical_full_interference(self):
        dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sim = feature_interference(dirs)
        assert abs(sim[0, 1] - 1.0) < 1e-5

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        dirs = rng.randn(10, 8)
        sim = feature_interference(dirs)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)


class TestDimensionalityAnalysis:
    def test_returns_expected_keys(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = dimensionality_analysis(model, seqs)
        assert "participation_ratio" in result
        assert "eigenvalue_spectra" in result
        assert "labels" in result

    def test_participation_ratio_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = dimensionality_analysis(model, seqs)
        # embed + 2 layers = 3
        assert result["participation_ratio"].shape == (3,)

    def test_participation_ratio_positive(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = dimensionality_analysis(model, seqs)
        assert np.all(result["participation_ratio"] >= 0)

    def test_eigenvalue_spectra_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = dimensionality_analysis(model, seqs)
        assert len(result["eigenvalue_spectra"]) == 3


class TestActivationCovariance:
    def test_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        cov = activation_covariance(model, seqs, "blocks.0.hook_resid_post")
        assert cov.shape == (model.cfg.d_model, model.cfg.d_model)

    def test_symmetric(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        cov = activation_covariance(model, seqs, "blocks.0.hook_resid_post")
        np.testing.assert_allclose(cov, cov.T, atol=1e-5)

    def test_empty_sequences(self):
        model = _make_model()
        cov = activation_covariance(model, [], "blocks.0.hook_resid_post")
        assert cov.shape == (model.cfg.d_model, model.cfg.d_model)
        np.testing.assert_allclose(cov, 0.0)


class TestFeatureSparsity:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        result = feature_sparsity(activations)
        assert "l0_mean" in result
        assert "l0_fraction" in result
        assert "kurtosis_mean" in result
        assert "gini_mean" in result

    def test_dense_activations(self):
        # All values are large -> L0 should be high
        activations = np.ones((50, 10)) * 5.0
        result = feature_sparsity(activations, threshold=0.1)
        assert result["l0_fraction"] > 0.9

    def test_sparse_activations(self):
        # Most values are zero
        activations = np.zeros((50, 100))
        activations[:, 0] = 1.0  # Only one dimension active
        result = feature_sparsity(activations, threshold=0.1)
        assert result["l0_fraction"] < 0.1

    def test_l0_range(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        result = feature_sparsity(activations)
        assert 0.0 <= result["l0_fraction"] <= 1.0
