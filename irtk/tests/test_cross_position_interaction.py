"""Tests for cross_position_interaction module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.cross_position_interaction import (
    pairwise_position_influence,
    directional_information_transfer,
    position_importance_ranking,
    interaction_clustering,
    critical_information_path,
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


class TestPairwisePositionInfluence:
    def test_basic(self, model, tokens):
        result = pairwise_position_influence(model, tokens, layer=0)
        assert "influence_matrix" in result
        assert "strongest_pairs" in result
        assert "most_influential_source" in result

    def test_matrix_shape(self, model, tokens):
        result = pairwise_position_influence(model, tokens, layer=0)
        assert result["influence_matrix"].shape == (len(tokens), len(tokens))


class TestDirectionalInformationTransfer:
    def test_basic(self, model, tokens):
        result = directional_information_transfer(model, tokens, source_pos=0, target_pos=-1)
        assert "per_layer" in result
        assert "cumulative_transfer" in result
        assert "peak_transfer_layer" in result

    def test_per_layer_populated(self, model, tokens):
        result = directional_information_transfer(model, tokens, source_pos=0, target_pos=-1)
        assert len(result["per_layer"]) == model.cfg.n_layers


class TestPositionImportanceRanking:
    def test_basic(self, model, tokens):
        result = position_importance_ranking(model, tokens)
        assert "position_scores" in result
        assert "ranked_positions" in result

    def test_scores_shape(self, model, tokens):
        result = position_importance_ranking(model, tokens)
        assert result["position_scores"].shape == (len(tokens),)


class TestInteractionClustering:
    def test_basic(self, model, tokens):
        result = interaction_clustering(model, tokens, layer=0, n_clusters=2)
        assert "cluster_assignments" in result
        assert "cluster_sizes" in result

    def test_cluster_count(self, model, tokens):
        result = interaction_clustering(model, tokens, layer=0, n_clusters=2)
        assert sum(result["cluster_sizes"]) == len(tokens)


class TestCriticalInformationPath:
    def test_basic(self, model, tokens):
        result = critical_information_path(model, tokens)
        assert "paths" in result
        assert "critical_positions" in result
        assert "n_active_paths" in result

    def test_has_paths(self, model, tokens):
        result = critical_information_path(model, tokens, threshold=0.05)
        assert result["n_active_paths"] > 0
