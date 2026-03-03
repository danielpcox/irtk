"""Tests for embedding space geometry."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_space_geometry import (
    embedding_isotropy, embedding_nearest_neighbors,
    embed_unembed_alignment, embedding_effective_dimension,
    embedding_geometry_summary,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def test_isotropy_structure(model):
    result = embedding_isotropy(model)
    assert "isotropy_score" in result
    assert "mean_cosine" in result
    assert "std_cosine" in result


def test_isotropy_values(model):
    result = embedding_isotropy(model)
    assert 0 <= result["isotropy_score"] <= 1.0
    assert result["std_cosine"] >= 0


def test_nearest_neighbors_structure(model):
    result = embedding_nearest_neighbors(model, token_id=5, top_k=3)
    assert "query_token" in result
    assert "neighbors" in result
    assert result["query_token"] == 5
    assert len(result["neighbors"]) == 3


def test_nearest_neighbors_no_self(model):
    result = embedding_nearest_neighbors(model, token_id=5, top_k=3)
    for tok, sim in result["neighbors"]:
        assert tok != 5


def test_embed_unembed_alignment(model):
    result = embed_unembed_alignment(model, top_k=3)
    assert "mean_alignment" in result
    assert "most_aligned" in result
    assert len(result["most_aligned"]) == 3


def test_embed_unembed_values(model):
    result = embed_unembed_alignment(model, top_k=3)
    for tok, cos in result["most_aligned"]:
        assert -1.0 <= cos <= 1.0


def test_effective_dimension_structure(model):
    result = embedding_effective_dimension(model)
    assert "effective_rank" in result
    assert "d_model" in result
    assert "utilization_ratio" in result


def test_effective_dimension_values(model):
    result = embedding_effective_dimension(model)
    assert result["effective_rank"] > 0
    assert 0 < result["utilization_ratio"] <= 1.0


def test_geometry_summary(model):
    result = embedding_geometry_summary(model)
    assert "isotropy" in result
    assert "effective_rank" in result
    assert "mean_eu_alignment" in result
