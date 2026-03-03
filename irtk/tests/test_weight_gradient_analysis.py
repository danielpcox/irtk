"""Tests for weight_gradient_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_gradient_analysis import (
    weight_sensitivity_profile, layer_weight_gradient_norms,
    attention_weight_gradients, mlp_weight_gradients,
    weight_gradient_summary,
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


def test_sensitivity_profile_structure(model, tokens):
    result = weight_sensitivity_profile(model, tokens)
    assert result["n_params"] > 0
    assert len(result["top_sensitive"]) <= 10


def test_sensitivity_profile_values(model, tokens):
    result = weight_sensitivity_profile(model, tokens)
    for p in result["top_sensitive"]:
        assert p["grad_norm"] >= 0
        assert p["param_norm"] > 0


def test_layer_gradient_norms_structure(model, tokens):
    result = layer_weight_gradient_norms(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["most_sensitive_layer"] < 2


def test_layer_gradient_fractions(model, tokens):
    result = layer_weight_gradient_norms(model, tokens)
    total = sum(p["fraction"] for p in result["per_layer"])
    assert abs(total - 1.0) < 0.01


def test_attention_weight_gradients(model, tokens):
    result = attention_weight_gradients(model, tokens, layer=0)
    assert "W_Q" in result["grad_norms"]
    assert "W_K" in result["grad_norms"]
    assert "W_V" in result["grad_norms"]
    assert "W_O" in result["grad_norms"]


def test_attention_weight_fractions(model, tokens):
    result = attention_weight_gradients(model, tokens, layer=0)
    total = sum(result["fractions"].values())
    assert abs(total - 1.0) < 0.01
    assert result["dominant_matrix"] in ("W_Q", "W_K", "W_V", "W_O")


def test_mlp_weight_gradients(model, tokens):
    result = mlp_weight_gradients(model, tokens, layer=0)
    assert "W_in" in result["grad_norms"]
    assert "W_out" in result["grad_norms"]
    assert result["dominant_matrix"] in ("W_in", "W_out")


def test_summary_structure(model, tokens):
    result = weight_gradient_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["most_sensitive_layer"] < 2


def test_summary_fractions(model, tokens):
    result = weight_gradient_summary(model, tokens)
    total = sum(p["fraction"] for p in result["per_layer"])
    assert abs(total - 1.0) < 0.01
