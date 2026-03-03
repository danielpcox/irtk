"""Tests for attention_pattern_evolution module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_evolution import (
    attention_stability_across_layers, attention_focus_evolution,
    head_agreement_evolution, attention_entropy_evolution,
    attention_evolution_summary,
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


def test_stability_structure(model, tokens):
    result = attention_stability_across_layers(model, tokens, head=0)
    assert len(result["per_pair"]) == 1  # 2 layers = 1 pair
    assert isinstance(result["is_stable"], bool)


def test_stability_range(model, tokens):
    result = attention_stability_across_layers(model, tokens, head=0)
    for p in result["per_pair"]:
        assert -1 <= p["cosine_similarity"] <= 1


def test_focus_evolution_structure(model, tokens):
    result = attention_focus_evolution(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["focus_shifts"] >= 0


def test_focus_evolution_entropy(model, tokens):
    result = attention_focus_evolution(model, tokens)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
        assert len(p["top_focus"]) > 0


def test_head_agreement_structure(model, tokens):
    result = head_agreement_evolution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["agreement_trend"] in ("increasing", "decreasing")


def test_head_agreement_range(model, tokens):
    result = head_agreement_evolution(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["mean_head_agreement"] <= 1


def test_entropy_evolution_structure(model, tokens):
    result = attention_entropy_evolution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["sharpens"], bool)


def test_entropy_evolution_heads(model, tokens):
    result = attention_entropy_evolution(model, tokens)
    for p in result["per_layer"]:
        assert len(p["per_head_entropy"]) == 4
        assert p["mean_entropy"] >= 0


def test_evolution_summary_structure(model, tokens):
    result = attention_evolution_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["mean_entropy"] >= 0
        assert 0 <= p["self_attention"] <= 1
