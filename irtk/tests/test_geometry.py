"""Tests for representation geometry analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.geometry import (
    representational_similarity,
    subspace_overlap,
    intrinsic_dimensionality,
    layer_similarity_matrix,
    representation_drift,
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


def _make_sequences():
    return [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]


class TestRepresentationalSimilarity:
    def test_self_similarity_cka(self):
        model = _make_model()
        seqs = _make_sequences()
        sim = representational_similarity(
            model, seqs, "blocks.0.hook_resid_post", "blocks.0.hook_resid_post"
        )
        assert abs(sim - 1.0) < 1e-4

    def test_range(self):
        model = _make_model()
        seqs = _make_sequences()
        sim = representational_similarity(
            model, seqs, "blocks.0.hook_resid_post", "blocks.1.hook_resid_post"
        )
        assert 0.0 <= sim <= 1.0 + 1e-5

    def test_cosine_method(self):
        model = _make_model()
        seqs = _make_sequences()
        sim = representational_similarity(
            model, seqs, "blocks.0.hook_resid_post", "blocks.1.hook_resid_post",
            method="cosine"
        )
        assert -1.0 <= sim <= 1.0 + 1e-5

    def test_invalid_method(self):
        model = _make_model()
        seqs = _make_sequences()
        with pytest.raises(ValueError, match="Unknown method"):
            representational_similarity(
                model, seqs, "blocks.0.hook_resid_post", "blocks.1.hook_resid_post",
                method="invalid"
            )


class TestSubspaceOverlap:
    def test_self_overlap(self):
        model = _make_model()
        seqs = _make_sequences()
        overlap = subspace_overlap(
            model, seqs, "blocks.0.hook_resid_post", "blocks.0.hook_resid_post",
            n_dims=3
        )
        assert abs(overlap - 1.0) < 1e-4

    def test_range(self):
        model = _make_model()
        seqs = _make_sequences()
        overlap = subspace_overlap(
            model, seqs, "blocks.0.hook_resid_post", "blocks.1.hook_resid_post",
            n_dims=3
        )
        assert 0.0 <= overlap <= 1.0 + 1e-5

    def test_empty_sequences(self):
        model = _make_model()
        overlap = subspace_overlap(
            model, [], "blocks.0.hook_resid_post", "blocks.1.hook_resid_post"
        )
        assert overlap == 0.0


class TestIntrinsicDimensionality:
    def test_participation_ratio(self):
        rng = np.random.RandomState(42)
        # Data along 3 dimensions
        activations = rng.randn(100, 3) @ rng.randn(3, 16)
        dim = intrinsic_dimensionality(activations, method="participation_ratio")
        assert 1.0 <= dim <= 5.0  # should be close to 3

    def test_explained_variance(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 3) @ rng.randn(3, 16)
        dim = intrinsic_dimensionality(activations, method="explained_variance_90")
        assert 1.0 <= dim <= 5.0

    def test_high_dimensional(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)  # uniformly spread
        dim = intrinsic_dimensionality(activations)
        assert dim > 5.0  # should be relatively high

    def test_one_dimensional(self):
        rng = np.random.RandomState(42)
        direction = rng.randn(16)
        activations = rng.randn(100, 1) @ direction[None, :]
        dim = intrinsic_dimensionality(activations)
        assert abs(dim - 1.0) < 0.1

    def test_invalid_method(self):
        rng = np.random.RandomState(42)
        activations = rng.randn(100, 16)
        with pytest.raises(ValueError, match="Unknown method"):
            intrinsic_dimensionality(activations, method="invalid")


class TestLayerSimilarityMatrix:
    def test_matrix_shape(self):
        model = _make_model()
        seqs = _make_sequences()
        result = layer_similarity_matrix(model, seqs)
        # embed + 2 layers = 3
        assert result["matrix"].shape == (3, 3)

    def test_diagonal_ones(self):
        model = _make_model()
        seqs = _make_sequences()
        result = layer_similarity_matrix(model, seqs)
        np.testing.assert_allclose(np.diag(result["matrix"]), 1.0, atol=1e-4)

    def test_symmetric(self):
        model = _make_model()
        seqs = _make_sequences()
        result = layer_similarity_matrix(model, seqs)
        np.testing.assert_allclose(result["matrix"], result["matrix"].T, atol=1e-5)

    def test_labels(self):
        model = _make_model()
        seqs = _make_sequences()
        result = layer_similarity_matrix(model, seqs)
        assert result["labels"] == ["embed", "block_0", "block_1"]


class TestRepresentationDrift:
    def test_returns_expected_keys(self):
        model = _make_model()
        seqs = _make_sequences()
        result = representation_drift(model, seqs)
        assert "l2_distances" in result
        assert "cosine_similarities" in result
        assert "labels" in result

    def test_shapes(self):
        model = _make_model()
        seqs = _make_sequences()
        result = representation_drift(model, seqs)
        # 2 transitions: embed->block_0, block_0->block_1
        assert result["l2_distances"].shape == (2,)
        assert result["cosine_similarities"].shape == (2,)
        assert len(result["labels"]) == 2

    def test_l2_nonnegative(self):
        model = _make_model()
        seqs = _make_sequences()
        result = representation_drift(model, seqs)
        assert np.all(result["l2_distances"] >= 0)

    def test_cosine_range(self):
        model = _make_model()
        seqs = _make_sequences()
        result = representation_drift(model, seqs)
        assert np.all(result["cosine_similarities"] >= -1.0 - 1e-5)
        assert np.all(result["cosine_similarities"] <= 1.0 + 1e-5)
