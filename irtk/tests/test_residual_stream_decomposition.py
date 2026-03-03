"""Tests for residual stream decomposition."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_decomposition import (
    component_contribution_norms, cumulative_component_balance,
    residual_component_cosines, residual_projection_decomposition,
    residual_stream_decomposition_summary,
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


def test_contribution_norms_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_contribution_norms(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_contribution_norms_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_contribution_norms(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["attn_norm"] >= 0
        assert p["mlp_norm"] >= 0
        assert p["residual_norm"] > 0


def test_cumulative_balance_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_component_balance(model, tokens, position=-1)
    assert "per_layer" in result
    assert "embed_norm" in result
    assert len(result["per_layer"]) == 2


def test_cumulative_balance_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_component_balance(model, tokens, position=-1)
    for p in result["per_layer"]:
        total = p["embed_fraction"] + p["attn_fraction"] + p["mlp_fraction"]
        assert abs(total - 1.0) < 0.01


def test_component_cosines_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_component_cosines(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_component_cosines_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_component_cosines(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert -1.0 <= p["attn_cosine"] <= 1.0
        assert -1.0 <= p["mlp_cosine"] <= 1.0


def test_projection_decomposition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_projection_decomposition(model, tokens, position=-1)
    assert "per_layer" in result
    assert "target_token" in result
    assert len(result["per_layer"]) == 2


def test_projection_decomposition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_projection_decomposition(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert 0 <= p["projection_fraction"] <= 1.0


def test_decomposition_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_stream_decomposition_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "attn_norm" in p
        assert "mlp_cosine" in p
