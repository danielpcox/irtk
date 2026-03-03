"""Tests for attention_composition_paths module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_composition_paths import (
    two_hop_attention, attention_path_strength,
    composition_score_matrix, attention_chain_strength,
    attention_composition_summary,
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


def test_two_hop_structure(model, tokens):
    result = two_hop_attention(model, tokens, layer1=0, layer2=1)
    assert result["n_pairs"] == 16  # 4x4


def test_two_hop_similarity_range(model, tokens):
    result = two_hop_attention(model, tokens, layer1=0, layer2=1)
    for p in result["per_head_pair"]:
        assert -1.1 <= p["virtual_direct_similarity"] <= 1.1


def test_path_strength_structure(model, tokens):
    result = attention_path_strength(model, tokens, source=0, target=-1)
    assert len(result["per_layer"]) == 2


def test_path_strength_values(model, tokens):
    result = attention_path_strength(model, tokens, source=0, target=-1)
    for p in result["per_layer"]:
        assert 0 <= p["max_strength"] <= 1
        assert 0 <= p["strongest_head"] < 4


def test_composition_score_structure(model, tokens):
    result = composition_score_matrix(model, tokens, layer1=0, layer2=1)
    assert len(result["per_pair"]) == 16


def test_composition_score_values(model, tokens):
    result = composition_score_matrix(model, tokens, layer1=0, layer2=1)
    for p in result["per_pair"]:
        assert -1.1 <= p["composition_score"] <= 1.1


def test_chain_strength(model, tokens):
    result = attention_chain_strength(model, tokens, source=0, target=-1)
    assert len(result["max_per_layer"]) == 2
    assert result["chain_strength"] >= 0
    assert isinstance(result["is_strong_path"], bool)


def test_summary_structure(model, tokens):
    result = attention_composition_summary(model, tokens)
    assert len(result["per_layer_pair"]) == 1  # 2 layers -> 1 pair


def test_summary_fields(model, tokens):
    result = attention_composition_summary(model, tokens)
    for p in result["per_layer_pair"]:
        assert -1.1 <= p["max_composition"] <= 1.1
