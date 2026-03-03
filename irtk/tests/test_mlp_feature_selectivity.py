"""Tests for MLP feature selectivity."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_feature_selectivity import (
    neuron_activation_selectivity, neuron_peak_response,
    neuron_position_preference, neuron_output_direction,
    feature_selectivity_summary,
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


def test_neuron_activation_selectivity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_selectivity(model, tokens, layer=0, top_k=5)
    assert "layer" in result
    assert "most_selective" in result
    assert "most_broad" in result
    assert "mean_selectivity" in result
    assert len(result["most_selective"]) == 5


def test_neuron_activation_selectivity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_selectivity(model, tokens, layer=0)
    assert 0 <= result["mean_selectivity"] <= 1.0
    for n in result["most_selective"]:
        assert 0 <= n["selectivity"] <= 1.0


def test_neuron_peak_response_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_peak_response(model, tokens, layer=0, top_k=5)
    assert "layer" in result
    assert "top_neurons" in result
    assert "mean_peak" in result
    assert "max_peak" in result
    assert len(result["top_neurons"]) == 5


def test_neuron_peak_response_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_peak_response(model, tokens, layer=0)
    assert result["max_peak"] >= result["mean_peak"]
    for n in result["top_neurons"]:
        assert n["peak_activation"] >= 0


def test_neuron_position_preference_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_position_preference(model, tokens, layer=0, neuron=0)
    assert "layer" in result
    assert "neuron" in result
    assert "per_position" in result
    assert "peak_position" in result
    assert len(result["per_position"]) == 5


def test_neuron_position_preference_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_position_preference(model, tokens, layer=0, neuron=0)
    assert 0 <= result["peak_position"] < 5
    assert result["n_active_positions"] >= 0


def test_neuron_output_direction_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_output_direction(model, layer=0, top_k=3)
    assert "layer" in result
    assert "per_neuron" in result
    assert len(result["per_neuron"]) == 3


def test_neuron_output_direction_content(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_output_direction(model, layer=0, top_k=3)
    for n in result["per_neuron"]:
        assert "promoted" in n
        assert "suppressed" in n
        assert len(n["promoted"]) == 5
        assert len(n["suppressed"]) == 5


def test_feature_selectivity_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = feature_selectivity_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_selectivity" in p
        assert "mean_peak" in p
