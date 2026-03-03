"""Tests for attention_residual_contribution module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_residual_contribution import (
    attention_residual_alignment, per_head_residual_contribution,
    attention_update_magnitude, attention_direction_consistency,
    attention_residual_summary,
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


def test_alignment_structure(model, tokens):
    result = attention_residual_alignment(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_reinforcing"], bool)


def test_alignment_cosine_range(model, tokens):
    result = attention_residual_alignment(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.1 <= p["cosine"] <= 1.1
        assert p["attn_norm"] >= 0


def test_per_head_structure(model, tokens):
    result = per_head_residual_contribution(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert 0 <= result["dominant_head"] < 4


def test_per_head_fractions(model, tokens):
    result = per_head_residual_contribution(model, tokens, layer=0)
    total = sum(h["fraction"] for h in result["per_head"])
    assert abs(total - 1.0) < 0.01


def test_update_magnitude_structure(model, tokens):
    result = attention_update_magnitude(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert result["mean_update_ratio"] >= 0


def test_update_magnitude_nonneg(model, tokens):
    result = attention_update_magnitude(model, tokens, layer=0)
    for p in result["per_position"]:
        assert p["update_ratio"] >= 0
        assert p["attn_norm"] >= 0


def test_direction_consistency(model, tokens):
    result = attention_direction_consistency(model, tokens, layer=0)
    assert -1.1 <= result["mean_direction_similarity"] <= 1.1
    assert isinstance(result["is_consistent"], bool)


def test_summary_structure(model, tokens):
    result = attention_residual_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = attention_residual_summary(model, tokens)
    for p in result["per_layer"]:
        assert isinstance(p["is_reinforcing"], bool)
        assert p["mean_update_ratio"] >= 0
