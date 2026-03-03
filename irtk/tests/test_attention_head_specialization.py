"""Tests for attention head specialization."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_specialization import (
    previous_token_score, induction_score,
    positional_head_score, head_entropy_profile,
    head_specialization_summary,
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


def test_previous_token_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = previous_token_score(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_previous_token" in result
    assert len(result["per_head"]) == 4


def test_previous_token_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = previous_token_score(model, tokens, layer=0)
    for p in result["per_head"]:
        assert 0 <= p["previous_token_score"] <= 1.0


def test_induction_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = induction_score(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_induction" in result
    assert len(result["per_head"]) == 4


def test_induction_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = induction_score(model, tokens, layer=0)
    for p in result["per_head"]:
        assert p["induction_score"] >= 0


def test_positional_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_head_score(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_positional" in result
    assert len(result["per_head"]) == 4


def test_positional_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_head_score(model, tokens, layer=0)
    for p in result["per_head"]:
        assert 0 <= p["positional_score"] <= 1.0


def test_entropy_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_entropy_profile(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    for p in result["per_head"]:
        assert p["entropy"] >= 0
        assert p["classification"] in ("focused", "diffuse")


def test_entropy_normalized(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_entropy_profile(model, tokens, layer=0)
    for p in result["per_head"]:
        assert 0 <= p["normalized_entropy"] <= 2.0  # generous bound


def test_specialization_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_specialization_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "n_previous_token" in p
        assert "n_induction" in p
        assert "n_positional" in p
