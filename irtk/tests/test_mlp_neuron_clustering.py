"""Tests for mlp_neuron_clustering module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_neuron_clustering import (
    neuron_activation_similarity, neuron_activity_profile,
    neuron_coactivation, neuron_output_direction_clustering,
    neuron_clustering_summary,
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
    return jnp.array([1, 5, 10, 15, 20])


def test_activation_similarity_structure(model, tokens):
    result = neuron_activation_similarity(model, tokens, layer=0)
    assert -1 <= result["mean_similarity"] <= 1
    assert isinstance(result["is_diverse"], bool)


def test_activation_similarity_neurons(model, tokens):
    result = neuron_activation_similarity(model, tokens, layer=0)
    assert result["n_neurons"] > 0


def test_activity_profile_structure(model, tokens):
    result = neuron_activity_profile(model, tokens, layer=0, top_k=5)
    assert len(result["top_active"]) > 0
    assert 0 <= result["mean_sparsity"] <= 1


def test_activity_profile_dead(model, tokens):
    result = neuron_activity_profile(model, tokens, layer=0)
    assert result["n_dead"] >= 0


def test_coactivation_structure(model, tokens):
    result = neuron_coactivation(model, tokens, layer=0, top_k=3)
    assert len(result["top_pairs"]) > 0
    assert result["mean_coactivation"] >= 0


def test_coactivation_rate_range(model, tokens):
    result = neuron_coactivation(model, tokens, layer=0)
    for p in result["top_pairs"]:
        assert 0 <= p["coactivation_rate"] <= 1


def test_output_direction_clustering(model):
    result = neuron_output_direction_clustering(model, layer=0)
    assert -1 <= result["mean_direction_similarity"] <= 1
    assert isinstance(result["is_clustered"], bool)


def test_summary_structure(model, tokens):
    result = neuron_clustering_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = neuron_clustering_summary(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["mean_sparsity"] <= 1
        assert p["n_dead"] >= 0
