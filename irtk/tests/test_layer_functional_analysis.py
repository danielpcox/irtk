"""Tests for layer_functional_analysis module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_functional_analysis import (
    layer_information_gain, layer_transformation_magnitude,
    layer_specialization_score, layer_prediction_contribution,
    layer_functional_summary,
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

def test_info_gain_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_gain(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_info_gain_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_gain(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
    # First layer has 0 KL (no previous)
    assert result["per_layer"][0]["kl_from_previous"] == 0.0

def test_transformation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_transformation_magnitude(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_transformation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_transformation_magnitude(model, tokens)
    for p in result["per_layer"]:
        assert p["delta_norm"] >= 0
        assert -1.0 <= p["pre_post_cosine"] <= 1.0

def test_specialization_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_specialization_score(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_specialization_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_specialization_score(model, tokens)
    for p in result["per_layer"]:
        assert abs(p["attn_fraction"] + p["mlp_fraction"] - 1.0) < 0.01
        assert p["specialization"] in ("attn_dominant", "mlp_dominant", "balanced")

def test_prediction_contribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_contribution(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_prediction_contribution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_contribution(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["logit_delta_norm"] >= 0
        assert isinstance(p["changed"], bool)

def test_functional_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_functional_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "entropy" in p
        assert "delta_norm" in p
        assert "specialization" in p
