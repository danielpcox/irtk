"""Tests for latent_space_navigation module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.latent_space_navigation import (
    representation_interpolation,
    latent_arithmetic,
    boundary_detection,
    manifold_exploration,
    latent_distance_map,
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
def tokens_a():
    return jnp.array([0, 5, 10, 15, 20])


@pytest.fixture
def tokens_b():
    return jnp.array([25, 30, 35, 40, 45])


class TestRepresentationInterpolation:
    def test_basic(self, model, tokens_a, tokens_b):
        result = representation_interpolation(model, tokens_a, tokens_b, n_steps=5)
        assert "interpolation_path" in result
        assert "top_predictions" in result
        assert "prediction_entropy" in result
        assert "total_path_length" in result

    def test_path_shape(self, model, tokens_a, tokens_b):
        result = representation_interpolation(model, tokens_a, tokens_b, n_steps=5)
        assert result["interpolation_path"].shape == (5, model.cfg.d_model)

    def test_predictions_count(self, model, tokens_a, tokens_b):
        result = representation_interpolation(model, tokens_a, tokens_b, n_steps=8)
        assert len(result["top_predictions"]) == 8


class TestLatentArithmetic:
    def test_basic(self, model, tokens_a, tokens_b):
        tokens_c = jnp.array([1, 2, 3, 4, 5])
        result = latent_arithmetic(model, tokens_a, tokens_b, tokens_c)
        assert "result_vector" in result
        assert "top_predictions" in result
        assert "cosine_to_a" in result

    def test_cosine_range(self, model, tokens_a, tokens_b):
        tokens_c = jnp.array([1, 2, 3, 4, 5])
        result = latent_arithmetic(model, tokens_a, tokens_b, tokens_c)
        assert -1.01 <= result["cosine_to_a"] <= 1.01


class TestBoundaryDetection:
    def test_basic(self, model, tokens_a, tokens_b):
        result = boundary_detection(model, tokens_a, tokens_b, n_probes=10)
        assert "boundary_alphas" in result
        assert "n_boundaries" in result
        assert "prediction_sequence" in result
        assert "confidences" in result

    def test_predictions_count(self, model, tokens_a, tokens_b):
        result = boundary_detection(model, tokens_a, tokens_b, n_probes=10)
        assert len(result["prediction_sequence"]) == 10


class TestManifoldExploration:
    def test_basic(self, model, tokens_a):
        result = manifold_exploration(model, tokens_a, n_directions=3, n_steps=2)
        assert "center_prediction" in result
        assert "direction_effects" in result
        assert "prediction_stability" in result

    def test_stability_range(self, model, tokens_a):
        result = manifold_exploration(model, tokens_a, n_directions=3, n_steps=2)
        assert 0 <= result["prediction_stability"] <= 1


class TestLatentDistanceMap:
    def test_basic(self, model, tokens_a, tokens_b):
        tokens_list = [tokens_a, tokens_b, jnp.array([1, 2, 3, 4, 5])]
        result = latent_distance_map(model, tokens_list)
        assert "distance_matrix" in result
        assert "cosine_similarity_matrix" in result
        assert "nearest_neighbors" in result

    def test_shape(self, model, tokens_a, tokens_b):
        tokens_list = [tokens_a, tokens_b]
        result = latent_distance_map(model, tokens_list)
        assert result["distance_matrix"].shape == (2, 2)
        assert result["cosine_similarity_matrix"].shape == (2, 2)

    def test_diagonal_zero(self, model, tokens_a, tokens_b):
        tokens_list = [tokens_a, tokens_b]
        result = latent_distance_map(model, tokens_list)
        assert float(result["distance_matrix"][0, 0]) < 1e-6
