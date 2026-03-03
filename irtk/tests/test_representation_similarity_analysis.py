"""Tests for representation_similarity_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.representation_similarity_analysis import (
    layer_representation_similarity, position_representation_similarity,
    representation_drift, representation_effective_dimension,
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


def test_layer_similarity_cosine(model, tokens):
    result = layer_representation_similarity(model, tokens, metric="cosine")
    assert result["similarity_matrix"].shape == (2, 2)
    assert result["n_layers"] == 2


def test_layer_similarity_cka(model, tokens):
    result = layer_representation_similarity(model, tokens, metric="cka")
    assert result["similarity_matrix"].shape == (2, 2)
    assert result["metric"] == "cka"


def test_position_similarity_structure(model, tokens):
    result = position_representation_similarity(model, tokens, layer=0)
    assert result["seq_len"] == 5
    assert isinstance(result["is_position_diverse"], bool)


def test_position_similarity_range(model, tokens):
    result = position_representation_similarity(model, tokens, layer=0)
    assert -1 <= result["mean_pairwise_similarity"] <= 1


def test_representation_drift_structure(model, tokens):
    result = representation_drift(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_gradual"], bool)


def test_representation_drift_cosines(model, tokens):
    result = representation_drift(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["cosine_to_final"] <= 1


def test_effective_dimension_structure(model, tokens):
    result = representation_effective_dimension(model, tokens, layer=0)
    assert result["participation_ratio"] > 0
    assert result["dim_for_90_pct"] >= 1


def test_effective_dimension_singular(model, tokens):
    result = representation_effective_dimension(model, tokens, layer=0)
    assert result["top_singular_value"] > 0


def test_geometry_summary_structure(model, tokens):
    result = representation_geometry_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["geometry_trend"] in ("expanding", "contracting")
    for p in result["per_layer"]:
        assert p["mean_norm"] > 0
