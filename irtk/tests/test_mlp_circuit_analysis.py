"""Tests for mlp_circuit_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_circuit_analysis import (
    neuron_to_logit_paths,
    mlp_knowledge_vs_feature,
    mlp_contribution_decomposition,
    mlp_nonlinearity_effect,
    mlp_layer_comparison,
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


def test_neuron_to_logit_paths(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_to_logit_paths(model, tokens, layer=0, top_k=3)
    assert len(result['neurons']) == 3
    for n in result['neurons']:
        assert len(n['top_promoted']) == 3
        assert n['total_abs_logit'] >= 0


def test_mlp_knowledge_vs_feature(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_knowledge_vs_feature(model, tokens, layer=0)
    assert result['n_knowledge'] + result['n_feature'] > 0
    assert 0 <= result['knowledge_fraction'] <= 1.0


def test_mlp_contribution_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_decomposition(model, tokens, layer=0)
    assert len(result['top_neurons']) > 0
    assert result['total_contribution'] >= 0
    assert result['n_neurons'] > 0


def test_mlp_nonlinearity_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_nonlinearity_effect(model, tokens, layer=0)
    assert result['pre_norm'] >= 0
    assert result['post_norm'] >= 0
    assert -1.0 <= result['pre_post_cosine'] <= 1.01


def test_mlp_layer_comparison(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_comparison(model, tokens)
    assert len(result['per_layer']) == 2
    for layer in result['per_layer']:
        assert layer['output_norm'] >= 0
        assert 0 <= layer['activation_sparsity'] <= 1.0


def test_neuron_logit_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_to_logit_paths(model, tokens, layer=0, top_k=5)
    for i in range(len(result['neurons']) - 1):
        assert result['neurons'][i]['total_abs_logit'] >= result['neurons'][i+1]['total_abs_logit'] - 0.01


def test_contribution_gini(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_decomposition(model, tokens, layer=0)
    assert -1.0 <= result['gini'] <= 1.0


def test_nonlinearity_sparsity(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_nonlinearity_effect(model, tokens, layer=0)
    assert 0 <= result['pre_near_zero'] <= 1.0
    assert 0 <= result['post_near_zero'] <= 1.0


def test_layer_comparison_most_active(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_comparison(model, tokens)
    assert 0 <= result['most_active_layer'] < 2
