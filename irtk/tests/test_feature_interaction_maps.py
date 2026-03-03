"""Tests for feature_interaction_maps module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.feature_interaction_maps import (
    feature_coactivation,
    mutual_suppression_enhancement,
    feature_clustering,
    interaction_strength,
    feature_dependency_graph,
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
def tokens_list():
    return [
        jnp.array([0, 5, 10, 15, 20]),
        jnp.array([1, 6, 11, 16, 21]),
        jnp.array([2, 7, 12, 17, 22]),
        jnp.array([3, 8, 13, 18, 23]),
    ]


def metric_fn(logits, tokens):
    return jnp.mean(logits[-1])


class TestFeatureCoactivation:
    def test_basic(self, model, tokens_list):
        result = feature_coactivation(model, tokens_list)
        assert "coactivation_matrix" in result
        assert "correlation_matrix" in result
        assert "top_pairs" in result

    def test_shapes(self, model, tokens_list):
        result = feature_coactivation(model, tokens_list)
        d = model.cfg.d_model
        assert result["coactivation_matrix"].shape == (d, d)
        assert result["correlation_matrix"].shape == (d, d)

    def test_top_pairs_count(self, model, tokens_list):
        result = feature_coactivation(model, tokens_list, top_k=3)
        assert len(result["top_pairs"]) <= 3


class TestMutualSuppressionEnhancement:
    def test_basic(self, model, tokens_list):
        result = mutual_suppression_enhancement(model, tokens_list, metric_fn, n_directions=3)
        assert "interaction_matrix" in result
        assert "single_effects" in result
        assert "suppressive_pairs" in result
        assert "enhancing_pairs" in result

    def test_matrix_shape(self, model, tokens_list):
        n = 3
        result = mutual_suppression_enhancement(model, tokens_list, metric_fn, n_directions=n)
        assert result["interaction_matrix"].shape == (n, n)


class TestFeatureClustering:
    def test_basic(self, model, tokens_list):
        result = feature_clustering(model, tokens_list, n_clusters=3)
        assert "cluster_assignments" in result
        assert "cluster_sizes" in result
        assert "within_cluster_correlation" in result

    def test_assignment_shape(self, model, tokens_list):
        result = feature_clustering(model, tokens_list, n_clusters=3)
        assert result["cluster_assignments"].shape == (model.cfg.d_model,)

    def test_cluster_sizes_sum(self, model, tokens_list):
        result = feature_clustering(model, tokens_list, n_clusters=3)
        assert sum(result["cluster_sizes"]) == model.cfg.d_model


class TestInteractionStrength:
    def test_basic(self, model, tokens_list):
        result = interaction_strength(model, tokens_list, metric_fn,
                                      components=[(0, 0), (0, 1)])
        assert "interaction_scores" in result
        assert "single_effects" in result
        assert "synergistic_pairs" in result
        assert "redundant_pairs" in result

    def test_matrix_shape(self, model, tokens_list):
        comps = [(0, 0), (0, 1), (1, 0)]
        result = interaction_strength(model, tokens_list, metric_fn, components=comps)
        assert result["interaction_scores"].shape == (3, 3)


class TestFeatureDependencyGraph:
    def test_basic(self, model, tokens_list):
        result = feature_dependency_graph(model, tokens_list, threshold=0.1)
        assert "adjacency_matrix" in result
        assert "edges" in result
        assert "node_degrees" in result
        assert "hub_dimensions" in result

    def test_adjacency_shape(self, model, tokens_list):
        result = feature_dependency_graph(model, tokens_list)
        d = model.cfg.d_model
        assert result["adjacency_matrix"].shape == (d, d)

    def test_degrees_shape(self, model, tokens_list):
        result = feature_dependency_graph(model, tokens_list)
        assert result["node_degrees"].shape == (model.cfg.d_model,)
