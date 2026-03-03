"""Tests for layer computation budget."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_computation_budget import (
    layer_norm_budget, layer_information_gain,
    component_balance, residual_growth_budget,
    computation_budget_summary,
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


def test_norm_budget_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_norm_budget(model, tokens)
    assert "per_layer" in result
    assert "total_budget" in result
    assert len(result["per_layer"]) == 2


def test_norm_budget_fractions_sum(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_norm_budget(model, tokens)
    total_frac = sum(p["fraction"] for p in result["per_layer"])
    assert abs(total_frac - 1.0) < 1e-4


def test_information_gain_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_gain(model, tokens, position=-1)
    assert "per_layer" in result
    assert "most_informative" in result
    assert len(result["per_layer"]) == 2


def test_information_gain_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_gain(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
    assert result["per_layer"][0]["info_gain"] == 0.0  # First layer has no prior


def test_component_balance_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_balance(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_component_balance_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_balance(model, tokens)
    for p in result["per_layer"]:
        assert abs(p["attn_fraction"] + p["mlp_fraction"] - 1.0) < 1e-4
        assert p["dominant"] in ("attn", "mlp")


def test_residual_growth_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_growth_budget(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_residual_growth_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_growth_budget(model, tokens)
    for p in result["per_layer"]:
        assert p["pre_norm"] > 0
        assert p["post_norm"] > 0
        assert p["growth_rate"] > 0


def test_computation_budget_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_budget_summary(model, tokens)
    assert "per_layer" in result
    assert "total_budget" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "dominant" in p
        assert "fraction" in p
