"""Tests for attention_causal_structure module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_causal_structure import (
    causal_attention_chain, attention_information_bottleneck,
    multi_hop_attention_paths, causal_influence_matrix,
    causal_structure_summary,
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


def test_causal_chain_structure(model, tokens):
    result = causal_attention_chain(model, tokens, target_position=-1)
    assert len(result["per_layer"]) == 2
    assert result["n_root_positions"] > 0


def test_causal_chain_target(model, tokens):
    result = causal_attention_chain(model, tokens, target_position=3)
    assert result["target_position"] == 3


def test_bottleneck_structure(model, tokens):
    result = attention_information_bottleneck(model, tokens)
    assert len(result["per_position"]) == 5
    assert isinstance(result["has_clear_bottleneck"], bool)


def test_bottleneck_scores(model, tokens):
    result = attention_information_bottleneck(model, tokens)
    for p in result["per_position"]:
        assert p["bottleneck_score"] >= 0
        assert p["total_incoming"] >= 0


def test_multi_hop_structure(model, tokens):
    result = multi_hop_attention_paths(model, tokens, source=0, target=4)
    assert len(result["per_depth"]) == 2
    assert result["source"] == 0
    assert result["target"] == 4


def test_multi_hop_strength(model, tokens):
    result = multi_hop_attention_paths(model, tokens, source=0, target=4)
    assert result["max_path_strength"] >= 0
    assert 1 <= result["best_depth"] <= 2


def test_influence_matrix_structure(model, tokens):
    result = causal_influence_matrix(model, tokens, layer=0)
    assert result["influence_matrix"].shape == (5, 5)
    assert result["mean_influence"] >= 0


def test_influence_matrix_max(model, tokens):
    result = causal_influence_matrix(model, tokens, layer=0)
    assert result["max_influence"] >= result["mean_influence"]


def test_summary_structure(model, tokens):
    result = causal_structure_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["overall_pattern"] in ("concentrated", "distributed")
    for p in result["per_layer"]:
        assert p["mean_entropy"] >= 0
        assert 0 <= p["bos_dominant_fraction"] <= 1
