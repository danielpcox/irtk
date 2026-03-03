"""Tests for mlp_memory_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_memory_analysis import (
    mlp_key_value_decomposition,
    mlp_retrieval_pattern,
    mlp_storage_capacity,
    mlp_input_selectivity,
    mlp_write_read_alignment,
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
    tokens = jnp.array([1, 10, 20, 30, 40])
    return model, tokens


def test_mlp_key_value_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_key_value_decomposition(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    assert 0 <= result['mean_active_fraction'] <= 1.0
    for p in result['per_position']:
        assert p['n_active_neurons'] >= 0


def test_mlp_retrieval_pattern(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_retrieval_pattern(model, tokens, layer=0, top_k=5)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['output_norm'] >= 0
        assert len(p['top_promoted']) == 5


def test_mlp_storage_capacity(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_storage_capacity(model, tokens, layer=0)
    assert result['n_neurons'] > 0
    assert 0 <= result['utilization'] <= 1.0
    assert result['effective_rank'] >= 1.0


def test_mlp_input_selectivity(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_selectivity(model, tokens, layer=0, top_k=5)
    assert len(result['per_neuron']) == 5
    for n in result['per_neuron']:
        assert 0 <= n['selectivity'] <= 1.01


def test_mlp_write_read_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_write_read_alignment(model, tokens, layer=0)
    assert result['source_layer'] == 0
    assert len(result['per_target']) == 1  # only layer 1 reads from layer 0


def test_decomposition_fraction_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_key_value_decomposition(model, tokens, layer=0)
    for p in result['per_position']:
        assert 0 <= p['active_fraction'] <= 1.0


def test_retrieval_mean_norm(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_retrieval_pattern(model, tokens, layer=0)
    assert result['mean_output_norm'] >= 0


def test_capacity_reuse(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_storage_capacity(model, tokens, layer=0)
    assert result['mean_neuron_reuse'] >= 0


def test_selectivity_counts(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_input_selectivity(model, tokens, layer=0, top_k=5)
    assert result['n_selective'] >= 0
