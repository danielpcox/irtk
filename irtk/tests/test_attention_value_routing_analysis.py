"""Tests for attention value routing analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_value_routing_analysis import (
    value_source_decomposition, value_diversity_per_head,
    attention_routing_entropy, value_output_alignment,
    value_routing_summary,
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


def test_value_source_decomposition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_source_decomposition(model, tokens, layer=0, head=0, position=-1)
    assert "per_source" in result
    assert "dominant_source" in result
    assert len(result["per_source"]) == 5


def test_value_source_decomposition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_source_decomposition(model, tokens, layer=0, head=0, position=-1)
    for s in result["per_source"]:
        assert 0 <= s["attention_weight"] <= 1.0
        assert s["contribution_norm"] >= 0


def test_value_diversity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_diversity_per_head(model, tokens, layer=0)
    assert "per_head" in result
    assert "mean_diversity" in result
    assert len(result["per_head"]) == 4


def test_value_diversity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_diversity_per_head(model, tokens, layer=0)
    for h in result["per_head"]:
        assert -1.0 <= h["mean_similarity"] <= 1.0


def test_routing_entropy_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_routing_entropy(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_focused" in result
    assert len(result["per_head"]) == 4


def test_routing_entropy_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_routing_entropy(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_entropy"] >= 0
        assert 0 <= h["normalized_entropy"] <= 1.5  # Can slightly exceed 1 due to mean


def test_value_output_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_output_alignment(model, tokens, layer=0, position=-1)
    assert "per_head" in result
    assert len(result["per_head"]) == 4


def test_value_output_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_output_alignment(model, tokens, layer=0, position=-1)
    for h in result["per_head"]:
        assert -1.0 <= h["cosine"] <= 1.0
        assert h["actual_norm"] >= 0


def test_value_routing_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_routing_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_diversity" in p
        assert "n_focused" in p
