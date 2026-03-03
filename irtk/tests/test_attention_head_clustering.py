"""Tests for attention head clustering."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_clustering import (
    head_pattern_similarity, head_output_direction_clustering,
    head_functional_fingerprint, head_redundancy_score,
    attention_head_clustering_summary,
)


@pytest.fixture
def model_and_tokens():
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
    model = jax.tree.unflatten(treedef, new_leaves)
    tokens = jnp.array([1, 5, 10, 15, 20])
    return model, tokens


def test_pattern_similarity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens, layer=0)
    assert "similarity_matrix" in result
    assert "per_pair" in result
    assert result["similarity_matrix"].shape == (4, 4)


def test_pattern_similarity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens, layer=0)
    for p in result["per_pair"]:
        assert -1.0 <= p["similarity"] <= 1.0


def test_pattern_similarity_diagonal(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens, layer=0)
    for i in range(4):
        assert abs(float(result["similarity_matrix"][i, i]) - 1.0) < 1e-4


def test_output_direction_clustering(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_direction_clustering(model, tokens, layer=0)
    assert result["similarity_matrix"].shape == (4, 4)
    assert len(result["per_pair"]) == 6  # C(4,2)


def test_functional_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_functional_fingerprint(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    for p in result["per_head"]:
        assert "entropy" in p
        assert "max_weight" in p
        assert "diagonal_score" in p
        assert p["entropy"] >= 0


def test_redundancy_score_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_redundancy_score(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_redundant" in result
    assert len(result["per_head"]) == 4


def test_redundancy_score_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_redundancy_score(model, tokens, layer=0)
    for p in result["per_head"]:
        assert -1.0 <= p["max_similarity"] <= 1.0
        assert p["most_similar_to"] != p["head"]


def test_redundancy_most_similar_valid(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_redundancy_score(model, tokens, layer=0)
    for p in result["per_head"]:
        assert 0 <= p["most_similar_to"] < 4


def test_clustering_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_head_clustering_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_pattern_similarity" in p
        assert "n_redundant" in p
