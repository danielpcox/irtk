"""Tests for mlp_neuron_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_neuron_profiling import (
    neuron_activation_profile,
    neuron_logit_impact,
    neuron_selectivity,
    neuron_correlation_clusters,
    dead_neuron_analysis,
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


def test_neuron_activation_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_profile(model, tokens, layer=0, top_k=5)
    assert len(result['top_neurons']) == 5
    assert result['n_active'] >= 0
    assert result['n_total'] > 0


def test_neuron_logit_impact(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_impact(model, tokens, layer=0, top_k=5)
    assert len(result['top_neurons']) == 5
    assert 0 <= result['target_token'] < 50


def test_neuron_selectivity(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_selectivity(model, tokens, layer=0, top_k=3)
    assert len(result['most_selective']) == 3
    assert len(result['least_selective']) == 3
    assert 0 <= result['mean_selectivity'] <= 1.0


def test_neuron_correlation_clusters(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_correlation_clusters(model, tokens, layer=0, threshold=0.5)
    assert result['n_active'] >= 0
    for p in result['correlated_pairs']:
        assert abs(p['correlation']) > 0.5


def test_dead_neuron_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = dead_neuron_analysis(model, tokens, layer=0)
    assert result['n_dead'] >= 0
    assert result['n_dead'] + result['n_near_dead'] + result['n_healthy'] == result['total_neurons']
    assert 0 <= result['dead_fraction'] <= 1.0


def test_activation_profile_has_activations(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_profile(model, tokens, layer=0, top_k=3)
    for n in result['top_neurons']:
        assert len(n['activations']) == 5  # 5 tokens


def test_logit_impact_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_impact(model, tokens, layer=0, top_k=5)
    for i in range(len(result['top_neurons']) - 1):
        assert abs(result['top_neurons'][i]['logit_contribution']) >= abs(result['top_neurons'][i+1]['logit_contribution']) - 0.01


def test_selectivity_ordering(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_selectivity(model, tokens, layer=0, top_k=3)
    # Most selective should have higher selectivity than least selective
    if result['most_selective'] and result['least_selective']:
        assert result['most_selective'][0]['selectivity'] >= result['least_selective'][0]['selectivity']


def test_dead_analysis_layer1(model_and_tokens):
    model, tokens = model_and_tokens
    result = dead_neuron_analysis(model, tokens, layer=1)
    assert result['total_neurons'] > 0
