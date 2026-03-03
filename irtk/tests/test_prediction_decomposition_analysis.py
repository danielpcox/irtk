"""Tests for prediction_decomposition_analysis module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_decomposition_analysis import (
    logit_contribution_by_layer, direct_logit_attribution,
    prediction_entropy_decomposition, component_prediction_agreement,
    prediction_decomposition_summary,
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

def test_logit_contribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_contribution_by_layer(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    assert "top_tokens" in result
    assert len(result["top_tokens"]) == 5

def test_logit_contribution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_contribution_by_layer(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["attn_contribution"] >= 0
        assert p["mlp_contribution"] >= 0

def test_direct_attribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = direct_logit_attribution(model, tokens, position=-1)
    assert "target_token" in result
    assert "per_component" in result
    assert len(result["per_component"]) == 2 * 2 + 2  # 2 layers * 2 (attn+mlp) + embed + bias

def test_direct_attribution_sums(model_and_tokens):
    model, tokens = model_and_tokens
    result = direct_logit_attribution(model, tokens, position=-1)
    # Total should be close to the actual logit
    assert abs(result["total_logit"]) > 0  # nonzero

def test_entropy_decomp_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_entropy_decomposition(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    assert "total_entropy_reduction" in result

def test_entropy_decomp_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_entropy_decomposition(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0

def test_agreement_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_prediction_agreement(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_agreement_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_prediction_agreement(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert 0 <= p["top_k_overlap"] <= 1.0
        assert -1.0 <= p["logit_cosine"] <= 1.0

def test_decomposition_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_decomposition_summary(model, tokens, position=-1)
    assert "top_tokens" in result
    assert "total_entropy_reduction" in result
    assert "per_layer" in result
