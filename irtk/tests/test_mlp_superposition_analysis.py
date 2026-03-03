"""Tests for mlp_superposition_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_superposition_analysis import (
    mlp_input_interference, mlp_output_interference,
    mlp_feature_capacity, mlp_neuron_orthogonality,
    mlp_superposition_summary,
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


def test_mlp_input_interference_structure(model):
    result = mlp_input_interference(model, layer=0)
    assert result['mean_interference'] >= 0
    assert result['max_interference'] >= result['mean_interference']
    assert isinstance(result['has_superposition'], bool)


def test_mlp_input_interference_pairs(model):
    result = mlp_input_interference(model, layer=0)
    assert result['n_high_interference_pairs'] >= 0
    assert result['d_mlp'] > 0


def test_mlp_output_interference_structure(model):
    result = mlp_output_interference(model, layer=0)
    assert result['mean_interference'] >= 0
    assert isinstance(result['has_superposition'], bool)


def test_mlp_feature_capacity_structure(model):
    result = mlp_feature_capacity(model, layer=0)
    assert result['capacity_ratio'] > 0
    assert result['effective_rank'] > 0
    assert result['rank_utilization'] > 0


def test_mlp_feature_capacity_ratio(model):
    result = mlp_feature_capacity(model, layer=0)
    assert result['d_mlp'] == result['theoretical_max_features']
    assert result['superposition_ratio'] > 0


def test_mlp_neuron_orthogonality_structure(model):
    result = mlp_neuron_orthogonality(model, layer=0, sample_size=20)
    assert result['input_mean_overlap'] >= 0
    assert result['output_mean_overlap'] >= 0
    assert isinstance(result['is_approximately_orthogonal'], bool)


def test_mlp_neuron_orthogonality_bounds(model):
    result = mlp_neuron_orthogonality(model, layer=0, sample_size=20)
    assert result['input_max_overlap'] >= result['input_mean_overlap']
    assert result['output_max_overlap'] >= result['output_mean_overlap']


def test_mlp_superposition_summary_structure(model):
    result = mlp_superposition_summary(model)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['superposition_level'] in ('high', 'moderate', 'low')
        assert p['capacity_ratio'] > 0


def test_mlp_superposition_summary_fields(model):
    result = mlp_superposition_summary(model)
    for p in result['per_layer']:
        assert p['mean_interference'] >= 0
        assert p['effective_rank'] > 0
