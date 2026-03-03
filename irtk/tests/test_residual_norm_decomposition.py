"""Tests for residual_norm_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_norm_decomposition import (
    norm_contribution_by_component, layerwise_norm_buildup,
    norm_direction_decomposition, cross_position_norm_profile,
    component_interference_matrix,
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


def test_norm_contribution_structure(model, tokens):
    result = norm_contribution_by_component(model, tokens)
    assert 'total_norm' in result
    assert len(result['per_component']) == 5  # embed + 2*(attn+mlp)
    assert result['total_norm'] > 0


def test_norm_contribution_fractions(model, tokens):
    result = norm_contribution_by_component(model, tokens)
    for c in result['per_component']:
        assert 'fraction_of_total' in c
        assert c['self_norm'] >= 0


def test_layerwise_norm_buildup_structure(model, tokens):
    result = layerwise_norm_buildup(model, tokens)
    assert result['final_norm'] > 0
    assert len(result['steps']) == 5  # embed + 2*(attn+mlp)
    assert result['steps'][0]['step'] == 'embed'


def test_layerwise_norm_buildup_monotonic_steps(model, tokens):
    result = layerwise_norm_buildup(model, tokens)
    for step in result['steps'][1:]:
        assert 'delta_norm' in step
        assert step['delta_norm'] >= 0


def test_norm_direction_decomposition_structure(model, tokens):
    result = norm_direction_decomposition(model, tokens)
    assert 'n_constructive' in result
    for c in result['per_component']:
        assert isinstance(c['is_constructive'], bool)
        assert 'cosine_with_residual' in c


def test_cross_position_norm_profile_structure(model, tokens):
    result = cross_position_norm_profile(model, tokens)
    assert len(result['per_position']) == 5
    assert 'mean_norm' in result
    for p in result['per_position']:
        assert isinstance(p['is_outlier'], bool)


def test_cross_position_norm_profile_z_scores(model, tokens):
    result = cross_position_norm_profile(model, tokens)
    for p in result['per_position']:
        assert 'z_score' in p
        assert p['norm'] > 0


def test_component_interference_matrix_structure(model, tokens):
    result = component_interference_matrix(model, tokens)
    n_components = 5  # embed + 2*(attn+mlp)
    expected_pairs = n_components * (n_components - 1) // 2
    assert len(result['pairs']) == expected_pairs


def test_component_interference_matrix_values(model, tokens):
    result = component_interference_matrix(model, tokens)
    for p in result['pairs']:
        assert -1.0 <= p['cosine'] <= 1.0
        assert 'dot_product' in p
