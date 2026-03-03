"""Tests for MLP weight spectrum."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_weight_spectrum import (
    mlp_input_weight_spectrum, mlp_output_weight_spectrum,
    mlp_in_out_alignment, mlp_neuron_norm_distribution,
    mlp_weight_spectrum_summary,
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


def test_input_spectrum_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_weight_spectrum(model, layer=0)
    assert "top_singular_values" in result
    assert "effective_rank" in result
    assert "condition_number" in result
    assert "total_energy" in result


def test_input_spectrum_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_weight_spectrum(model, layer=0)
    assert result["effective_rank"] >= 1.0
    assert result["condition_number"] >= 1.0
    assert result["total_energy"] > 0


def test_output_spectrum_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_weight_spectrum(model, layer=0)
    assert "top_singular_values" in result
    assert "effective_rank" in result


def test_output_spectrum_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_weight_spectrum(model, layer=0)
    assert result["effective_rank"] >= 1.0
    assert result["total_energy"] > 0


def test_in_out_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_in_out_alignment(model, layer=0)
    assert "product_effective_rank" in result
    assert "product_trace" in result
    assert "product_frobenius" in result
    assert "top_svs" in result


def test_in_out_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_in_out_alignment(model, layer=0)
    assert result["product_frobenius"] >= 0


def test_neuron_norm_distribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_norm_distribution(model, layer=0)
    assert "in_mean_norm" in result
    assert "out_mean_norm" in result
    assert "in_out_correlation" in result


def test_neuron_norm_distribution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_norm_distribution(model, layer=0)
    assert result["in_mean_norm"] > 0
    assert result["out_mean_norm"] > 0
    assert -1.0 <= result["in_out_correlation"] <= 1.0


def test_weight_spectrum_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_weight_spectrum_summary(model)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "in_rank" in p
        assert "out_rank" in p
