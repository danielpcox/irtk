"""Tests for attention_key_value_dynamics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_key_value_dynamics import (
    key_norm_evolution, value_norm_evolution,
    key_value_alignment, key_similarity_across_positions,
    kv_dynamics_summary,
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


def test_key_norm_evolution_structure(model, tokens):
    result = key_norm_evolution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["norm_trend"] in ("increasing", "decreasing", "stable")


def test_key_norm_positive(model, tokens):
    result = key_norm_evolution(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_key_norm"] >= 0


def test_value_norm_evolution_structure(model, tokens):
    result = value_norm_evolution(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["mean_value_norm"] >= 0


def test_key_value_alignment_range(model, tokens):
    result = key_value_alignment(model, tokens, layer=0)
    assert -1 <= result["mean_alignment"] <= 1
    assert isinstance(result["is_aligned"], bool)


def test_key_value_alignment_per_head(model, tokens):
    result = key_value_alignment(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    for h in result["per_head"]:
        assert -1 <= h["mean_alignment"] <= 1


def test_key_similarity_structure(model, tokens):
    result = key_similarity_across_positions(model, tokens, layer=0, head=0)
    assert -1 <= result["mean_key_similarity"] <= 1
    assert isinstance(result["is_diverse"], bool)


def test_key_similarity_matrix_shape(model, tokens):
    result = key_similarity_across_positions(model, tokens, layer=0, head=0)
    assert result["similarity_matrix_shape"] == [5, 5]


def test_kv_summary_structure(model, tokens):
    result = kv_dynamics_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_kv_summary_fields(model, tokens):
    result = kv_dynamics_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_key_norm"] >= 0
        assert p["mean_value_norm"] >= 0
        assert "mean_kv_alignment" in p
        assert p["kv_norm_ratio"] >= 0
