"""Tests for cross-component interaction."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.cross_component_interaction import (
    attn_mlp_cosine, attn_mlp_logit_agreement,
    component_contribution_to_prediction, residual_mid_analysis,
    cross_component_summary,
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


def test_cosine_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_mlp_cosine(model, tokens, layer=0, position=-1)
    assert "cosine" in result
    assert "relationship" in result
    assert result["relationship"] in ("reinforcing", "competing", "orthogonal")


def test_cosine_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_mlp_cosine(model, tokens, layer=0, position=-1)
    assert -1.0 <= result["cosine"] <= 1.0
    assert result["attn_norm"] >= 0
    assert result["mlp_norm"] >= 0


def test_logit_agreement_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_mlp_logit_agreement(model, tokens, layer=0, position=-1, top_k=5)
    assert "attn_promoted" in result
    assert "mlp_promoted" in result
    assert "n_overlap" in result


def test_logit_agreement_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_mlp_logit_agreement(model, tokens, layer=0, position=-1, top_k=5)
    assert 0 <= result["n_overlap"] <= 5
    assert 0 <= result["agreement_fraction"] <= 1.0


def test_contribution_to_prediction(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_contribution_to_prediction(model, tokens, layer=0, position=-1)
    assert "predicted_token" in result
    assert "attn_contribution" in result
    assert "mlp_contribution" in result


def test_contribution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_contribution_to_prediction(model, tokens, layer=0, position=-1)
    assert 0 <= result["predicted_token"] < 50


def test_residual_mid_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_mid_analysis(model, tokens, layer=0, position=-1)
    assert "pre_norm" in result
    assert "mid_norm" in result
    assert "post_norm" in result


def test_residual_mid_cosines(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_mid_analysis(model, tokens, layer=0, position=-1)
    assert -1.0 <= result["pre_mid_cosine"] <= 1.0
    assert -1.0 <= result["mid_post_cosine"] <= 1.0


def test_cross_component_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_component_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "cosine" in p
        assert "relationship" in p
        assert "agreement" in p
