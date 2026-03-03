"""Tests for attention_head_diversity module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_diversity import (
    pattern_diversity, output_diversity,
    entropy_diversity, attention_focus_diversity,
    head_diversity_summary,
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


def test_pattern_diversity_structure(model, tokens):
    result = pattern_diversity(model, tokens, layer=0)
    assert -1 <= result["mean_pattern_similarity"] <= 1
    assert isinstance(result["is_diverse"], bool)


def test_pattern_diversity_pairs(model, tokens):
    result = pattern_diversity(model, tokens, layer=0)
    assert len(result["head_pairs"]) == 6  # C(4,2) = 6


def test_output_diversity_structure(model, tokens):
    result = output_diversity(model, tokens, layer=0, position=-1)
    assert -1 <= result["mean_output_similarity"] <= 1
    assert isinstance(result["is_diverse"], bool)


def test_entropy_diversity_structure(model, tokens):
    result = entropy_diversity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["entropy_range"] >= 0


def test_entropy_diversity_values(model, tokens):
    result = entropy_diversity(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_entropy"] >= 0


def test_focus_diversity_structure(model, tokens):
    result = attention_focus_diversity(model, tokens, layer=0)
    assert len(result["per_query"]) == 5
    assert 0 <= result["focus_diversity"] <= 1


def test_focus_unique(model, tokens):
    result = attention_focus_diversity(model, tokens, layer=0)
    for a in result["per_query"]:
        assert 1 <= a["n_unique_focuses"] <= 4


def test_summary_structure(model, tokens):
    result = head_diversity_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = head_diversity_summary(model, tokens)
    for p in result["per_layer"]:
        assert isinstance(p["is_pattern_diverse"], bool)
        assert isinstance(p["is_entropy_diverse"], bool)
