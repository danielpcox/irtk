"""Tests for attention_position_encoding module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_position_encoding import (
    positional_attention_bias, position_sensitivity,
    relative_position_preference, position_encoding_strength,
    position_encoding_summary,
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


def test_positional_bias_structure(model, tokens):
    result = positional_attention_bias(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["n_positional_heads"] >= 0


def test_positional_bias_values(model, tokens):
    result = positional_attention_bias(model, tokens, layer=0)
    for h in result["per_head"]:
        assert -1.1 <= h["positional_correlation"] <= 1.1
        assert isinstance(h["is_positional"], bool)


def test_position_sensitivity_structure(model, tokens):
    result = position_sensitivity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert 0 <= result["mean_sensitivity"] <= 1


def test_position_sensitivity_values(model, tokens):
    result = position_sensitivity(model, tokens, layer=0)
    for h in result["per_head"]:
        assert -1.1 <= h["pattern_similarity"] <= 1.1
        assert isinstance(h["is_position_sensitive"], bool)


def test_relative_preference_structure(model, tokens):
    result = relative_position_preference(model, tokens, layer=0, head=0)
    assert len(result["per_distance"]) > 0
    assert isinstance(result["prefers_recent"], bool)


def test_relative_preference_values(model, tokens):
    result = relative_position_preference(model, tokens, layer=0, head=0)
    for d in result["per_distance"]:
        assert d["distance"] >= 0
        assert d["mean_attention"] >= 0


def test_encoding_strength_structure(model, tokens):
    result = position_encoding_strength(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["most_distinct_layer"] < 2


def test_summary_structure(model, tokens):
    result = position_encoding_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = position_encoding_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["n_positional_heads"] >= 0
        assert -1.1 <= p["position_similarity"] <= 1.1
