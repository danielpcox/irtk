"""Tests for layer_functional_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_functional_profiling import (
    layer_prediction_impact, layer_computation_type,
    layer_information_gain, layer_redundancy_analysis,
    layer_functional_summary,
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


def test_layer_prediction_impact_structure(model, tokens):
    result = layer_prediction_impact(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert isinstance(p['prediction_changed'], bool)
        assert p['kl_divergence'] >= 0


def test_layer_computation_type_structure(model, tokens):
    result = layer_computation_type(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['computation_type'] in ('attention_dominated', 'mlp_dominated', 'balanced')


def test_layer_computation_type_fractions(model, tokens):
    result = layer_computation_type(model, tokens)
    for p in result['per_layer']:
        assert abs(p['attn_fraction'] + (1 - p['attn_fraction']) - 1.0) < 0.01


def test_layer_information_gain_structure(model, tokens):
    result = layer_information_gain(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['perpendicular_component'] >= 0


def test_layer_information_gain_fraction(model, tokens):
    result = layer_information_gain(model, tokens)
    for p in result['per_layer']:
        assert 0 <= p['new_info_fraction'] <= 1.01


def test_layer_redundancy_analysis_structure(model, tokens):
    result = layer_redundancy_analysis(model, tokens)
    assert len(result['pairs']) == 1  # C(2,1)
    for p in result['pairs']:
        assert -1 <= p['similarity'] <= 1
        assert isinstance(p['is_redundant'], bool)


def test_layer_redundancy_count(model, tokens):
    result = layer_redundancy_analysis(model, tokens)
    assert result['n_redundant_pairs'] >= 0


def test_layer_functional_summary_structure(model, tokens):
    result = layer_functional_summary(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_magnitude'] >= 0
        assert p['mlp_magnitude'] >= 0


def test_layer_functional_summary_entropy(model, tokens):
    result = layer_functional_summary(model, tokens)
    for p in result['per_layer']:
        assert p['mean_attn_entropy'] >= 0
        assert p['logit_impact'] >= 0
