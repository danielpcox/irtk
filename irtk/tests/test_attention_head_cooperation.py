"""Tests for attention_head_cooperation module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_cooperation import (
    within_layer_cooperation, cross_layer_head_pipeline,
    head_output_diversity, head_contribution_ranking,
    cooperation_summary,
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


def test_within_layer_structure(model, tokens):
    result = within_layer_cooperation(model, tokens, layer=0)
    assert len(result["pairs"]) == 6  # C(4,2)
    assert result["n_redundant"] >= 0


def test_within_layer_relations(model, tokens):
    result = within_layer_cooperation(model, tokens, layer=0)
    for p in result["pairs"]:
        assert p["relation"] in ("redundant", "competing", "complementary", "independent")
        assert -1 <= p["output_cosine"] <= 1


def test_cross_layer_pipeline_structure(model, tokens):
    result = cross_layer_head_pipeline(model, tokens)
    assert len(result["per_transition"]) == 1  # 2 layers = 1 transition
    assert isinstance(result["has_pipeline"], bool)


def test_head_output_diversity_structure(model, tokens):
    result = head_output_diversity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert isinstance(result["is_diverse"], bool)


def test_head_output_diversity_range(model, tokens):
    result = head_output_diversity(model, tokens, layer=0)
    assert -1 <= result["mean_pairwise_cosine"] <= 1
    assert 0 <= result["mean_diversity"] <= 1


def test_contribution_ranking_structure(model, tokens):
    result = head_contribution_ranking(model, tokens)
    assert len(result["heads"]) == 8  # 2 layers * 4 heads


def test_contribution_ranking_sorted(model, tokens):
    result = head_contribution_ranking(model, tokens)
    ranges = [h["logit_range"] for h in result["heads"]]
    assert ranges == sorted(ranges, reverse=True)


def test_cooperation_summary_structure(model, tokens):
    result = cooperation_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert isinstance(p["is_diverse"], bool)
        assert p["n_redundant_pairs"] >= 0


def test_cooperation_summary_complementary(model, tokens):
    result = cooperation_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["n_complementary_pairs"] >= 0
