"""Tests for MLP gradient flow."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_gradient_flow import (
    mlp_input_gradient, mlp_weight_gradient_norms,
    mlp_neuron_gradient_profile, mlp_gradient_sparsity,
    mlp_gradient_flow_summary,
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


def test_mlp_input_gradient_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_gradient(model, tokens, layer=0)
    assert "layer" in result
    assert "per_position" in result
    assert "mean_grad_norm" in result
    assert len(result["per_position"]) == 5


def test_mlp_input_gradient_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_gradient(model, tokens, layer=0)
    for p in result["per_position"]:
        assert p["grad_norm"] >= 0
    assert result["mean_grad_norm"] >= 0


def test_weight_gradient_norms_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_weight_gradient_norms(model, tokens, layer=0)
    assert "W_in_grad_norm" in result
    assert "W_out_grad_norm" in result
    assert "b_in_grad_norm" in result
    assert "b_out_grad_norm" in result
    assert "total_grad_norm" in result


def test_weight_gradient_norms_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_weight_gradient_norms(model, tokens, layer=0)
    assert result["W_in_grad_norm"] >= 0
    assert result["W_out_grad_norm"] >= 0
    assert result["total_grad_norm"] > 0


def test_neuron_gradient_profile_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_gradient_profile(model, tokens, layer=0, top_k=5)
    assert "top_neurons" in result
    assert len(result["top_neurons"]) == 5
    assert "mean_neuron_grad" in result
    assert "max_neuron_grad" in result


def test_neuron_gradient_profile_ordered(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_gradient_profile(model, tokens, layer=0, top_k=5)
    grads = [n["grad_norm"] for n in result["top_neurons"]]
    assert grads == sorted(grads, reverse=True)


def test_gradient_sparsity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gradient_sparsity(model, tokens, layer=0)
    assert "n_significant" in result
    assert "total_neurons" in result
    assert "gradient_sparsity" in result


def test_gradient_sparsity_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gradient_sparsity(model, tokens, layer=0)
    assert 0 <= result["gradient_sparsity"] <= 1.0
    assert result["n_significant"] <= result["total_neurons"]


def test_gradient_flow_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gradient_flow_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "total_grad_norm" in p
        assert "gradient_sparsity" in p
