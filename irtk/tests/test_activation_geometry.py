"""Tests for activation_geometry module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_geometry import (
    activation_manifold_dimension,
    representation_similarity_across_inputs,
    activation_cluster_analysis,
    representation_curvature,
    activation_norm_distribution,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


@pytest.fixture
def tokens_list():
    return [
        jnp.array([0, 5, 10, 15, 20, 25, 30, 35]),
        jnp.array([1, 6, 11, 16, 21, 26, 31, 36]),
        jnp.array([2, 7, 12, 17, 22, 27, 32, 37]),
        jnp.array([3, 8, 13, 18, 23, 28, 33, 38]),
        jnp.array([4, 9, 14, 19, 24, 29, 34, 39]),
    ]


class TestActivationManifoldDimension:
    def test_output_keys(self, model, tokens_list):
        r = activation_manifold_dimension(model, tokens_list)
        assert "intrinsic_dim" in r
        assert "explained_variance" in r
        assert "cumulative_variance" in r
        assert "n_for_90pct" in r
        assert "n_for_99pct" in r

    def test_intrinsic_dim_positive(self, model, tokens_list):
        r = activation_manifold_dimension(model, tokens_list)
        assert r["intrinsic_dim"] > 0

    def test_variance_sums_to_one(self, model, tokens_list):
        r = activation_manifold_dimension(model, tokens_list)
        assert abs(np.sum(r["explained_variance"]) - 1.0) < 1e-5

    def test_cumulative_monotonic(self, model, tokens_list):
        r = activation_manifold_dimension(model, tokens_list)
        cv = r["cumulative_variance"]
        assert np.all(np.diff(cv) >= -1e-8)

    def test_single_input_fallback(self, model):
        r = activation_manifold_dimension(model, [jnp.array([0, 5, 10])])
        assert r["intrinsic_dim"] == float(model.cfg.d_model)


class TestRepresentationSimilarityAcrossInputs:
    def test_output_keys(self, model, tokens):
        tokens_b = jnp.array([1, 6, 11, 16, 21, 26, 31, 36])
        r = representation_similarity_across_inputs(model, tokens, tokens_b)
        assert "layer_similarities" in r
        assert "divergence_layer" in r
        assert "convergence_layer" in r
        assert "initial_similarity" in r
        assert "final_similarity" in r

    def test_shape(self, model, tokens):
        tokens_b = jnp.array([1, 6, 11, 16, 21, 26, 31, 36])
        r = representation_similarity_across_inputs(model, tokens, tokens_b)
        n_layers = model.cfg.n_layers
        assert r["layer_similarities"].shape == (n_layers + 1,)

    def test_similarity_bounded(self, model, tokens):
        tokens_b = jnp.array([1, 6, 11, 16, 21, 26, 31, 36])
        r = representation_similarity_across_inputs(model, tokens, tokens_b)
        assert np.all(r["layer_similarities"] >= -1.0 - 1e-5)
        assert np.all(r["layer_similarities"] <= 1.0 + 1e-5)

    def test_same_input_high_similarity(self, model, tokens):
        r = representation_similarity_across_inputs(model, tokens, tokens)
        assert np.all(r["layer_similarities"] > 0.99)


class TestActivationClusterAnalysis:
    def test_output_keys(self, model, tokens_list):
        r = activation_cluster_analysis(model, tokens_list, n_clusters=2)
        assert "cluster_assignments" in r
        assert "cluster_sizes" in r
        assert "within_cluster_similarity" in r
        assert "between_cluster_similarity" in r
        assert "cluster_centroids" in r

    def test_assignments_valid(self, model, tokens_list):
        r = activation_cluster_analysis(model, tokens_list, n_clusters=2)
        assert len(r["cluster_assignments"]) == len(tokens_list)
        assert np.all(r["cluster_assignments"] >= 0)
        assert np.all(r["cluster_assignments"] < 2)

    def test_sizes_sum(self, model, tokens_list):
        r = activation_cluster_analysis(model, tokens_list, n_clusters=2)
        assert np.sum(r["cluster_sizes"]) == len(tokens_list)

    def test_centroids_shape(self, model, tokens_list):
        r = activation_cluster_analysis(model, tokens_list, n_clusters=2)
        assert r["cluster_centroids"].shape == (2, model.cfg.d_model)


class TestRepresentationCurvature:
    def test_output_keys(self, model, tokens):
        r = representation_curvature(model, tokens)
        assert "curvatures" in r
        assert "max_curvature_layer" in r
        assert "mean_curvature" in r
        assert "trajectory_length" in r

    def test_curvatures_shape(self, model, tokens):
        r = representation_curvature(model, tokens)
        n_layers = model.cfg.n_layers
        assert r["curvatures"].shape == (max(0, n_layers - 1),)

    def test_curvatures_nonneg(self, model, tokens):
        r = representation_curvature(model, tokens)
        assert np.all(r["curvatures"] >= -1e-8)

    def test_trajectory_length_positive(self, model, tokens):
        r = representation_curvature(model, tokens)
        assert r["trajectory_length"] > 0


class TestActivationNormDistribution:
    def test_output_keys(self, model, tokens_list):
        r = activation_norm_distribution(model, tokens_list)
        assert "norms" in r
        assert "mean_norm" in r
        assert "std_norm" in r
        assert "min_norm" in r
        assert "max_norm" in r
        assert "coefficient_of_variation" in r

    def test_norms_shape(self, model, tokens_list):
        r = activation_norm_distribution(model, tokens_list)
        assert len(r["norms"]) == len(tokens_list)

    def test_norms_positive(self, model, tokens_list):
        r = activation_norm_distribution(model, tokens_list)
        assert np.all(r["norms"] > 0)

    def test_stats_consistent(self, model, tokens_list):
        r = activation_norm_distribution(model, tokens_list)
        assert r["min_norm"] <= r["mean_norm"] <= r["max_norm"]
