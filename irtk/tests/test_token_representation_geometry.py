"""Tests for token_representation_geometry module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_representation_geometry import (
    representation_clustering, representation_spread,
    representation_velocity, inter_token_distances,
    representation_geometry_summary,
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


def test_clustering_structure(model, tokens):
    result = representation_clustering(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_clustered"], bool)


def test_clustering_similarity_range(model, tokens):
    result = representation_clustering(model, tokens, layer=0)
    assert -1.1 <= result["mean_pairwise_similarity"] <= 1.1


def test_spread_structure(model, tokens):
    result = representation_spread(model, tokens, layer=0)
    assert result["effective_rank"] > 0
    assert result["total_variance"] >= 0
    assert 0 <= result["top_sv_fraction"] <= 1


def test_spread_rank(model, tokens):
    result = representation_spread(model, tokens, layer=0)
    assert isinstance(result["is_low_rank"], bool)


def test_velocity_structure(model, tokens):
    result = representation_velocity(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["mean_velocity"] >= 0


def test_velocity_values(model, tokens):
    result = representation_velocity(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["velocity"] >= 0
        assert p["relative_velocity"] >= 0


def test_distances_structure(model, tokens):
    result = inter_token_distances(model, tokens, layer=0)
    assert len(result["distances"]) == 10  # C(5,2)
    assert result["mean_distance"] >= 0


def test_summary_structure(model, tokens):
    result = representation_geometry_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = representation_geometry_summary(model, tokens)
    for p in result["per_layer"]:
        assert isinstance(p["is_clustered"], bool)
        assert p["effective_rank"] > 0
