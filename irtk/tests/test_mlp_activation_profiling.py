"""Tests for mlp_activation_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_activation_profiling import (
    mlp_pre_activation_profile, mlp_post_activation_profile,
    mlp_neuron_activation_distribution, mlp_activation_position_profile,
    mlp_pre_post_correlation,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([1, 5, 10, 15, 20])


def test_pre_activation_profile_structure(model, tokens):
    result = mlp_pre_activation_profile(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_norm'] > 0
        assert 'fraction_positive' in p


def test_pre_activation_profile_values(model, tokens):
    result = mlp_pre_activation_profile(model, tokens)
    for p in result['per_layer']:
        assert 0 <= p['fraction_positive'] <= 1
        assert p['max_activation'] > 0


def test_post_activation_profile_structure(model, tokens):
    result = mlp_post_activation_profile(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'sparsity' in p
        assert 0 <= p['sparsity'] <= 1


def test_neuron_activation_distribution_structure(model, tokens):
    result = mlp_neuron_activation_distribution(model, tokens, layer=0, top_k=5)
    assert result['layer'] == 0
    assert len(result['top_neurons']) == 5
    assert result['n_dead'] >= 0


def test_neuron_activation_distribution_dead_fraction(model, tokens):
    result = mlp_neuron_activation_distribution(model, tokens, layer=0)
    assert 0 <= result['dead_fraction'] <= 1
    assert result['d_mlp'] > 0


def test_activation_position_profile_structure(model, tokens):
    result = mlp_activation_position_profile(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['activation_norm'] >= 0
        assert 0 <= p['sparsity'] <= 1


def test_activation_position_profile_tokens(model, tokens):
    result = mlp_activation_position_profile(model, tokens, layer=0)
    for i, p in enumerate(result['per_position']):
        assert p['token'] == int(tokens[i])


def test_pre_post_correlation_structure(model, tokens):
    result = mlp_pre_post_correlation(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['norm_correlation'] <= 1
        assert p['compression_ratio'] > 0


def test_pre_post_correlation_norms(model, tokens):
    result = mlp_pre_post_correlation(model, tokens)
    for p in result['per_layer']:
        assert p['pre_mean_norm'] > 0
        assert p['post_mean_norm'] > 0
