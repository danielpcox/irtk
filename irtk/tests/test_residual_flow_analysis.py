"""Tests for residual_flow_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_flow_analysis import (
    residual_direction_flow, residual_norm_flow,
    residual_component_flow, residual_signal_noise,
    residual_cross_position_flow,
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


def test_residual_direction_flow_structure(model, tokens):
    result = residual_direction_flow(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['direction_cosine'] <= 1


def test_residual_direction_flow_drift(model, tokens):
    result = residual_direction_flow(model, tokens)
    assert -1 <= result['embed_to_final_cosine'] <= 1


def test_residual_norm_flow_structure(model, tokens):
    result = residual_norm_flow(model, tokens)
    assert result['embed_norm'] > 0
    assert result['final_norm'] > 0
    assert len(result['per_layer']) == 2


def test_residual_norm_flow_growth(model, tokens):
    result = residual_norm_flow(model, tokens)
    for p in result['per_layer']:
        assert p['growth_factor'] > 0
        assert p['cumulative_growth'] > 0


def test_residual_component_flow_structure(model, tokens):
    result = residual_component_flow(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0


def test_residual_component_flow_fractions(model, tokens):
    result = residual_component_flow(model, tokens)
    for p in result['per_layer']:
        assert abs(p['attn_fraction'] + p['mlp_fraction'] - 1.0) < 0.01


def test_residual_signal_noise_structure(model, tokens):
    result = residual_signal_noise(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['noise'] >= 0


def test_residual_cross_position_flow_structure(model, tokens):
    result = residual_cross_position_flow(model, tokens)
    assert len(result['per_position']) == 5
    assert isinstance(result['is_diverse'], bool)


def test_residual_cross_position_flow_similarity(model, tokens):
    result = residual_cross_position_flow(model, tokens)
    assert -1 <= result['mean_pairwise_similarity'] <= 1
