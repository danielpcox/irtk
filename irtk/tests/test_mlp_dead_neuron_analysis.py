"""Tests for MLP dead neuron analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_dead_neuron_analysis import (
    dead_neuron_detection, neuron_activation_frequency,
    neuron_activation_magnitude_distribution, near_dead_neurons,
    dead_neuron_summary,
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


def test_dead_neuron_detection_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = dead_neuron_detection(model, tokens, layer=0)
    assert "n_dead" in result
    assert "total_neurons" in result
    assert "dead_fraction" in result
    assert "dead_indices" in result


def test_dead_neuron_detection_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = dead_neuron_detection(model, tokens, layer=0)
    assert 0 <= result["dead_fraction"] <= 1.0
    assert result["n_dead"] <= result["total_neurons"]


def test_neuron_activation_frequency_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_frequency(model, tokens, layer=0)
    assert "mean_frequency" in result
    assert "median_frequency" in result
    assert "histogram" in result
    assert "n_always_active" in result
    assert "n_never_active" in result


def test_neuron_activation_frequency_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_frequency(model, tokens, layer=0)
    assert 0 <= result["mean_frequency"] <= 1.0
    assert 0 <= result["median_frequency"] <= 1.0


def test_magnitude_distribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_magnitude_distribution(model, tokens, layer=0)
    assert "global_mean" in result
    assert "global_max" in result
    assert "mean_std" in result
    assert "n_high_variance" in result


def test_magnitude_distribution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_magnitude_distribution(model, tokens, layer=0)
    assert result["global_max"] >= result["global_mean"]
    assert result["mean_std"] >= 0


def test_near_dead_neurons_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = near_dead_neurons(model, tokens, layer=0)
    assert "n_near_dead" in result
    assert "threshold" in result
    assert "weakest_neurons" in result
    assert len(result["weakest_neurons"]) <= 10


def test_near_dead_neurons_ordered(model_and_tokens):
    model, tokens = model_and_tokens
    result = near_dead_neurons(model, tokens, layer=0)
    acts = [n["mean_activation"] for n in result["weakest_neurons"]]
    assert acts == sorted(acts)


def test_dead_neuron_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = dead_neuron_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "dead_fraction" in p
        assert "mean_frequency" in p
