"""Tests for MLP bottleneck analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_bottleneck_analysis import (
    mlp_compression_ratio, mlp_hidden_utilization,
    mlp_input_reconstruction, mlp_expansion_selectivity,
    mlp_bottleneck_summary,
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


def test_compression_ratio_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_compression_ratio(model, tokens, layer=0)
    assert "input_effective_rank" in result
    assert "hidden_effective_rank" in result
    assert "compression_ratio" in result


def test_compression_ratio_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_compression_ratio(model, tokens, layer=0)
    assert result["input_effective_rank"] > 0
    assert result["hidden_effective_rank"] > 0
    assert result["compression_ratio"] > 0


def test_hidden_utilization_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_hidden_utilization(model, tokens, layer=0)
    assert "active_fraction" in result
    assert "n_active" in result
    assert "n_total" in result


def test_hidden_utilization_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_hidden_utilization(model, tokens, layer=0)
    assert 0 <= result["active_fraction"] <= 1.0
    assert result["n_active"] <= result["n_total"]


def test_input_reconstruction_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_reconstruction(model, tokens, layer=0)
    assert "per_position" in result
    assert "mean_reconstruction" in result
    assert len(result["per_position"]) == 5


def test_input_reconstruction_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_reconstruction(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.0 <= p["cosine"] <= 1.0


def test_expansion_selectivity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_expansion_selectivity(model, tokens, layer=0, top_k=3)
    assert "most_selective" in result
    assert "least_selective" in result
    assert len(result["most_selective"]) == 3


def test_expansion_selectivity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_expansion_selectivity(model, tokens, layer=0, top_k=3)
    for _, sel in result["most_selective"]:
        assert 0 <= sel <= 1.0
    assert 0 <= result["mean_selectivity"] <= 1.0


def test_bottleneck_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_bottleneck_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "compression_ratio" in p
        assert "active_fraction" in p
