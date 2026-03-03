"""Tests for vocabulary_space_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.vocabulary_space_analysis import (
    logit_space_neighbors, vocabulary_coverage,
    prediction_diversity_across_positions, token_logit_trajectory,
    vocabulary_space_summary,
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


def test_logit_neighbors_structure(model, tokens):
    result = logit_space_neighbors(model, tokens, position=-1, top_k=5)
    assert len(result["neighbors"]) == 5
    assert 0 <= result["top_prob"] <= 1


def test_logit_neighbors_sorted(model, tokens):
    result = logit_space_neighbors(model, tokens, top_k=5)
    logits = [n["logit"] for n in result["neighbors"]]
    assert logits == sorted(logits, reverse=True)


def test_vocabulary_coverage_structure(model, tokens):
    result = vocabulary_coverage(model, tokens, layer=-1)
    assert len(result["per_position"]) == 5
    assert result["mean_coverage"] >= 0


def test_vocabulary_coverage_range(model, tokens):
    result = vocabulary_coverage(model, tokens)
    for p in result["per_position"]:
        assert p["n_tokens_above_threshold"] >= 0
        assert p["entropy"] >= 0


def test_prediction_diversity_structure(model, tokens):
    result = prediction_diversity_across_positions(model, tokens)
    assert len(result["top_tokens"]) == 5
    assert isinstance(result["is_diverse"], bool)


def test_prediction_diversity_similarity(model, tokens):
    result = prediction_diversity_across_positions(model, tokens)
    assert -1 <= result["mean_logit_similarity"] <= 1


def test_token_logit_trajectory_structure(model, tokens):
    result = token_logit_trajectory(model, tokens, target_token=10)
    assert len(result["per_layer"]) == 2
    assert result["target_token"] == 10


def test_token_logit_trajectory_rank(model, tokens):
    result = token_logit_trajectory(model, tokens, target_token=10)
    for p in result["per_layer"]:
        assert p["target_rank"] >= 0
        assert 0 <= p["target_prob"] <= 1


def test_vocabulary_summary_structure(model, tokens):
    result = vocabulary_space_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["sharpening"], bool)
    for p in result["per_layer"]:
        assert p["mean_entropy"] >= 0
        assert 0 <= p["mean_max_prob"] <= 1
