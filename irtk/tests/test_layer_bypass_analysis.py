"""Tests for layer_bypass_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_bypass_analysis import (
    layer_contribution_vs_passthrough,
    effective_depth,
    skip_connection_utilization,
    shortcut_detection,
    minimal_circuit_depth,
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


def test_layer_contribution_vs_passthrough(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_contribution_vs_passthrough(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['passthrough_ratio'] >= 0
        assert p['contribution_ratio'] >= 0
        assert isinstance(p['is_bypass'], bool)


def test_effective_depth(model_and_tokens):
    model, tokens = model_and_tokens
    result = effective_depth(model, tokens)
    assert result['total_layers'] == 2
    assert 0 <= result['effective_depth'] <= 2
    assert 0 <= result['depth_ratio'] <= 1.0


def test_effective_depth_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    low = effective_depth(model, tokens, contribution_threshold=0.001)
    high = effective_depth(model, tokens, contribution_threshold=100.0)
    assert low['effective_depth'] >= high['effective_depth']


def test_skip_connection_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = skip_connection_utilization(model, tokens)
    assert -1.0 <= result['embedding_final_cosine'] <= 1.0
    assert result['embedding_norm'] > 0
    assert result['final_norm'] > 0
    assert len(result['per_layer_contribution']) == 2


def test_shortcut_detection(model_and_tokens):
    model, tokens = model_and_tokens
    result = shortcut_detection(model, tokens)
    assert 'target_token' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert isinstance(p['correct'], bool)
        assert 0 <= p['target_probability'] <= 1.0
    assert isinstance(result['has_shortcut'], bool)


def test_shortcut_detection_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = shortcut_detection(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_minimal_circuit_depth(model_and_tokens):
    model, tokens = model_and_tokens
    result = minimal_circuit_depth(model, tokens)
    assert 1 <= result['minimal_depth'] <= result['total_layers']
    assert len(result['per_layer']) == 2


def test_minimal_depth_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    low = minimal_circuit_depth(model, tokens, logit_threshold=0.1)
    high = minimal_circuit_depth(model, tokens, logit_threshold=0.99)
    assert low['minimal_depth'] <= high['minimal_depth']


def test_bypass_count(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_contribution_vs_passthrough(model, tokens)
    assert result['n_bypass'] == len(result['bypass_layers'])
