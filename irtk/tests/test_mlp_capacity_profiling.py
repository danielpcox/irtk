"""Tests for mlp_capacity_profiling module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_capacity_profiling import (
    mlp_dead_neuron_profile, mlp_activation_diversity,
    mlp_weight_utilization, mlp_information_throughput,
    mlp_capacity_summary,
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

def test_dead_neuron_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_dead_neuron_profile(model, tokens, layer=0)
    assert "n_dead" in result
    assert "dead_fraction" in result
    assert "n_total" in result

def test_dead_neuron_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_dead_neuron_profile(model, tokens, layer=0)
    assert 0 <= result["dead_fraction"] <= 1.0
    assert result["n_dead"] <= result["n_total"]
    assert result["mean_activity"] >= 0

def test_diversity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_diversity(model, tokens, layer=0)
    assert "mean_diversity" in result
    assert "most_diverse" in result
    assert "least_diverse" in result

def test_diversity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_diversity(model, tokens, layer=0)
    assert 0 <= result["mean_diversity"] <= 1.0
    assert len(result["most_diverse"]) == 5
    assert len(result["least_diverse"]) == 5

def test_weight_utilization_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_weight_utilization(model, layer=0)
    assert "w_in_effective_rank" in result
    assert "w_out_effective_rank" in result
    assert "w_in_utilization" in result

def test_weight_utilization_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_weight_utilization(model, layer=0)
    assert result["w_in_utilization"] > 0
    assert result["w_out_utilization"] > 0
    assert result["d_model"] == 16

def test_throughput_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_information_throughput(model, tokens, layer=0)
    assert "mean_throughput_ratio" in result
    assert "mean_residual_fraction" in result
    assert len(result["per_position"]) == len(tokens)

def test_throughput_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_information_throughput(model, tokens, layer=0)
    assert result["mean_throughput_ratio"] > 0
    for p in result["per_position"]:
        assert p["throughput_ratio"] >= 0

def test_capacity_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_capacity_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "dead_fraction" in p
        assert "diversity_score" in p
        assert "w_in_utilization" in p
