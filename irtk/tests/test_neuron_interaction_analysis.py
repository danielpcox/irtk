"""Tests for neuron_interaction_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.neuron_interaction_analysis import (
    neuron_coactivation_matrix,
    neuron_interference,
    cooperative_neuron_groups,
    neuron_compensation,
    neuron_ensemble_effect,
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


def test_neuron_coactivation_matrix(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_coactivation_matrix(model, tokens, layer=0)
    assert result['n_neurons_analyzed'] > 0
    assert result['mean_coactivation'] >= 0


def test_neuron_interference(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_interference(model, tokens, layer=0, top_k=3)
    assert len(result['per_neuron']) == 3
    for n in result['per_neuron']:
        assert n['activation'] >= 0


def test_cooperative_neuron_groups(model_and_tokens):
    model, tokens = model_and_tokens
    result = cooperative_neuron_groups(model, tokens, layer=0, threshold=0.3)
    assert result['n_groups'] >= 0
    assert result['n_neurons_analyzed'] > 0


def test_neuron_compensation(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_compensation(model, tokens, layer=0, neuron_idx=0)
    assert result['neuron_idx'] == 0
    assert result['max_logit_effect'] >= 0
    assert result['mean_logit_effect'] >= 0
    assert 0 <= result['compensation_ratio'] <= 1.0


def test_neuron_ensemble_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_ensemble_effect(model, tokens, layer=0, top_k=3)
    assert len(result['individual_effects']) == 3
    assert result['joint_effect'] >= 0
    assert result['synergy_ratio'] >= 0


def test_coactivation_symmetric(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_coactivation_matrix(model, tokens, layer=0)
    mat = result['coactivation_matrix']
    n = result['n_neurons_analyzed']
    for i in range(min(n, 5)):
        for j in range(min(n, 5)):
            assert abs(float(mat[i, j]) - float(mat[j, i])) < 0.01


def test_interference_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_interference(model, tokens, layer=0, top_k=5)
    for i in range(len(result['per_neuron']) - 1):
        assert result['per_neuron'][i]['total_interference'] >= result['per_neuron'][i+1]['total_interference'] - 0.01


def test_ensemble_sum_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_ensemble_effect(model, tokens, layer=0, top_k=3)
    assert result['sum_individual'] >= 0


def test_compensation_different_neurons(model_and_tokens):
    model, tokens = model_and_tokens
    r0 = neuron_compensation(model, tokens, layer=0, neuron_idx=0)
    r1 = neuron_compensation(model, tokens, layer=0, neuron_idx=1)
    assert r0['neuron_idx'] == 0
    assert r1['neuron_idx'] == 1
