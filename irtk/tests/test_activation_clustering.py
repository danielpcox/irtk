"""Tests for activation_clustering module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_clustering import (
    residual_stream_clustering,
    layer_activation_archetypes,
    activation_transition_analysis,
    position_clustering,
    component_output_clustering,
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


class TestResidualStreamClustering:
    def test_basic(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5]), jnp.array([10, 20, 30, 40, 49])]
        result = residual_stream_clustering(model, tokens_list, n_clusters=2)
        assert "cluster_assignments" in result
        assert "cluster_centers" in result
        assert "cluster_sizes" in result
        assert "within_cluster_variance" in result

    def test_shapes(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5]), jnp.array([10, 20, 30, 40, 49])]
        result = residual_stream_clustering(model, tokens_list, n_clusters=2)
        assert result["cluster_assignments"].shape == (3,)
        assert result["cluster_centers"].shape == (2, model.cfg.d_model)
        assert result["cluster_sizes"].shape == (2,)

    def test_assignments_valid(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5]), jnp.array([10, 20, 30, 40, 49])]
        result = residual_stream_clustering(model, tokens_list, n_clusters=2)
        assert np.all(result["cluster_assignments"] >= 0)
        assert np.all(result["cluster_assignments"] < 2)


class TestLayerActivationArchetypes:
    def test_basic(self, model, tokens):
        result = layer_activation_archetypes(model, tokens, n_archetypes=2)
        assert "archetypes" in result
        assert "layer_archetype_assignments" in result
        assert "archetype_prevalence" in result
        assert "layer_archetype_distribution" in result

    def test_shapes(self, model, tokens):
        result = layer_activation_archetypes(model, tokens, n_archetypes=2)
        nl = model.cfg.n_layers
        seq_len = len(tokens)
        assert result["archetypes"].shape == (2, model.cfg.d_model)
        assert result["layer_archetype_assignments"].shape == (nl, seq_len)
        assert result["archetype_prevalence"].shape == (2,)
        assert result["layer_archetype_distribution"].shape == (nl, 2)

    def test_prevalence_sums(self, model, tokens):
        result = layer_activation_archetypes(model, tokens, n_archetypes=2)
        assert abs(sum(result["archetype_prevalence"]) - 1.0) < 0.01


class TestActivationTransitionAnalysis:
    def test_basic(self, model, tokens):
        result = activation_transition_analysis(model, tokens)
        assert "transition_magnitudes" in result
        assert "transition_directions" in result
        assert "cosine_continuity" in result
        assert "mean_transition_magnitude" in result
        assert "smoothness" in result

    def test_shapes(self, model, tokens):
        result = activation_transition_analysis(model, tokens)
        nl = model.cfg.n_layers
        d = model.cfg.d_model
        assert result["transition_magnitudes"].shape == (nl - 1,)
        assert result["transition_directions"].shape == (nl - 1, d)

    def test_magnitudes_nonneg(self, model, tokens):
        result = activation_transition_analysis(model, tokens)
        assert np.all(result["transition_magnitudes"] >= 0)


class TestPositionClustering:
    def test_basic(self, model, tokens):
        result = position_clustering(model, tokens, n_clusters=2)
        assert "cluster_assignments" in result
        assert "cluster_centers" in result
        assert "cluster_sizes" in result
        assert "position_similarity" in result

    def test_shapes(self, model, tokens):
        result = position_clustering(model, tokens, n_clusters=2)
        seq_len = len(tokens)
        assert result["cluster_assignments"].shape == (seq_len,)
        assert result["cluster_centers"].shape == (2, model.cfg.d_model)
        assert result["position_similarity"].shape == (seq_len, seq_len)

    def test_similarity_range(self, model, tokens):
        result = position_clustering(model, tokens, n_clusters=2)
        assert np.all(result["position_similarity"] >= -1.01)
        assert np.all(result["position_similarity"] <= 1.01)


class TestComponentOutputClustering:
    def test_basic(self, model, tokens):
        result = component_output_clustering(model, tokens, n_clusters=2)
        assert "component_names" in result
        assert "cluster_assignments" in result
        assert "cluster_centers" in result
        assert "similarity_matrix" in result

    def test_component_count(self, model, tokens):
        result = component_output_clustering(model, tokens, n_clusters=2)
        nl = model.cfg.n_layers
        assert len(result["component_names"]) == nl * 2  # attn + mlp per layer

    def test_similarity_range(self, model, tokens):
        result = component_output_clustering(model, tokens, n_clusters=2)
        assert np.all(result["similarity_matrix"] >= -1.01)
        assert np.all(result["similarity_matrix"] <= 1.01)
