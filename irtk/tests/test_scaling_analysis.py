"""Tests for scaling_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.scaling_analysis import (
    layer_capacity_utilization,
    feature_density,
    component_saturation,
    depth_contribution_profile,
    representation_compression,
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


def test_layer_capacity_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_capacity_utilization(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'effective_rank' in p
        assert 'utilization' in p
        assert p['effective_rank'] > 0
        assert 0 <= p['utilization'] <= 1.0


def test_layer_capacity_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_capacity_utilization(model, tokens, layers=[0])
    assert len(result['per_layer']) == 1
    assert result['per_layer'][0]['layer'] == 0


def test_feature_density(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_list = [tokens, jnp.array([5, 15, 25, 35, 45])]
    result = feature_density(model, tokens_list, layer=-1, n_directions=20)
    assert 'density_estimate' in result
    assert 0 <= result['density_estimate'] <= 1.0
    assert result['n_directions'] == 20


def test_component_saturation(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_saturation(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'attn_mean_max' in p
        assert 'mlp_dead_fraction' in p
        assert 0 <= p['attn_mean_max'] <= 1.0
        assert 0 <= p['mlp_dead_fraction'] <= 1.0


def test_depth_contribution_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = depth_contribution_profile(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'residual_norm' in p
        assert 'attn_contribution_norm' in p
        assert 'mlp_contribution_norm' in p
        assert p['residual_norm'] >= 0
        assert p['attn_contribution_norm'] >= 0


def test_depth_contribution_with_pos(model_and_tokens):
    model, tokens = model_and_tokens
    result = depth_contribution_profile(model, tokens, pos=0)
    assert len(result['per_layer']) == 2
    assert result['total_attn_contribution'] >= 0


def test_representation_compression(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_list = [
        jnp.array([1, 10, 20, 30, 40]),
        jnp.array([5, 15, 25, 35, 45]),
        jnp.array([2, 12, 22, 32, 42]),
    ]
    result = representation_compression(model, tokens_list)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'effective_dimensionality' in p
        assert 'mean_pairwise_distance' in p
        assert p['effective_dimensionality'] > 0


def test_representation_compression_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_list = [tokens, jnp.array([5, 15, 25, 35, 45])]
    result = representation_compression(model, tokens_list, layers=[1])
    assert len(result['per_layer']) == 1
    assert result['per_layer'][0]['layer'] == 1


def test_saturation_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_saturation(model, tokens, layers=[0])
    assert len(result['per_layer']) == 1
