"""Tests for head_function_classifier module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_function_classifier import (
    classify_previous_token, classify_induction, classify_copying,
    classify_positional, head_function_summary,
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
    tokens = jnp.array([1, 5, 10, 5, 10, 20])
    return model, tokens

def test_previous_token_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_previous_token(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4

def test_previous_token_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_previous_token(model, tokens, layer=0)
    for h in result["per_head"]:
        assert 0 <= h["previous_token_score"] <= 1
        assert isinstance(h["is_previous_token"], bool)

def test_induction_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_induction(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4

def test_induction_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_induction(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["induction_score"] >= 0
        assert isinstance(h["is_induction"], bool)

def test_copying_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_copying(model, tokens, layer=0, top_k=3)
    assert "per_head" in result
    assert len(result["per_head"]) == 4

def test_copying_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_copying(model, tokens, layer=0, top_k=3)
    for h in result["per_head"]:
        assert "copy_score" in h
        assert len(h["top_copied_tokens"]) == 3

def test_positional_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_positional(model, tokens, layer=0)
    assert "per_head" in result
    for h in result["per_head"]:
        assert 0 <= h["positional_score"] <= 1

def test_positional_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_positional(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_pattern_variance"] >= 0

def test_function_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_function_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for layer_info in result["per_layer"]:
        assert "per_head" in layer_info
        assert len(layer_info["per_head"]) == 4
        for h in layer_info["per_head"]:
            assert "roles" in h
            assert len(h["roles"]) >= 1
