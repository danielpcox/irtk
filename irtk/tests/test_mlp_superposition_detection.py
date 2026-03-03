"""Tests for MLP superposition detection."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_superposition_detection import (
    neuron_activation_correlation, neuron_output_interference,
    neuron_polysemanticity, superposition_dimensionality,
    mlp_superposition_summary,
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


def test_correlation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_correlation(model, tokens, layer=0)
    assert "correlation_matrix" in result
    assert "mean_off_diagonal" in result
    assert "max_correlation" in result


def test_correlation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_correlation(model, tokens, layer=0)
    assert result["mean_off_diagonal"] >= 0
    assert result["max_correlation"] >= 0


def test_interference_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_output_interference(model, tokens, layer=0, top_k=3)
    assert "most_interfering" in result
    assert "mean_interference" in result
    assert len(result["most_interfering"]) == 3


def test_interference_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_output_interference(model, tokens, layer=0, top_k=3)
    for p in result["most_interfering"]:
        assert p["interference"] >= 0
    assert result["mean_interference"] >= 0


def test_polysemanticity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_polysemanticity(model, tokens, layer=0, top_k=3)
    assert "most_polysemantic" in result
    assert "mean_polysemanticity" in result
    assert len(result["most_polysemantic"]) == 3


def test_polysemanticity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_polysemanticity(model, tokens, layer=0, top_k=3)
    for _, score in result["most_polysemantic"]:
        assert 0 <= score <= 1.0
    assert 0 <= result["mean_polysemanticity"] <= 1.0


def test_dimensionality_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = superposition_dimensionality(model, tokens, layer=0)
    assert "effective_rank" in result
    assert "d_mlp" in result
    assert "superposition_ratio" in result


def test_dimensionality_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = superposition_dimensionality(model, tokens, layer=0)
    assert result["effective_rank"] > 0
    assert result["superposition_ratio"] >= 1.0


def test_superposition_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_superposition_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_correlation" in p
        assert "superposition_ratio" in p
