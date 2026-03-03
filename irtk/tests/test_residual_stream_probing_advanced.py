"""Tests for residual_stream_probing_advanced module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_probing_advanced import (
    token_identity_recovery, next_token_prediction_quality,
    positional_information_content, residual_feature_separability,
    residual_probing_summary,
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
    # Include repeated tokens for separability test
    tokens = jnp.array([1, 5, 10, 5, 10, 20])
    return model, tokens

def test_identity_recovery_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_identity_recovery(model, tokens, layer=0)
    assert "accuracy" in result
    assert "per_position" in result
    assert len(result["per_position"]) == len(tokens)

def test_identity_recovery_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_identity_recovery(model, tokens, layer=0)
    assert 0 <= result["accuracy"] <= 1.0
    for p in result["per_position"]:
        assert isinstance(p["correct"], bool)

def test_prediction_quality_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = next_token_prediction_quality(model, tokens, layer=0)
    assert "accuracy" in result
    assert "mean_rank" in result
    assert len(result["per_position"]) == len(tokens) - 1

def test_prediction_quality_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = next_token_prediction_quality(model, tokens, layer=0)
    assert 0 <= result["accuracy"] <= 1.0
    assert result["mean_rank"] >= 0

def test_positional_content_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_information_content(model, tokens, layer=0)
    assert "content_score" in result
    assert "positional_score" in result
    assert result["dominant"] in ("content", "position", "mixed")

def test_positional_content_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_information_content(model, tokens, layer=0)
    assert result["n_same_token_pairs"] >= 0
    assert result["n_diff_token_pairs"] >= 0

def test_separability_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_feature_separability(model, tokens, layer=0)
    assert "separability_score" in result
    assert "n_unique_tokens" in result

def test_separability_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_feature_separability(model, tokens, layer=0)
    assert result["separability_score"] >= 0
    assert result["n_unique_tokens"] == 4  # 1,5,10,20

def test_probing_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_probing_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "identity_accuracy" in p
        assert "prediction_accuracy" in p
