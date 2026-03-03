"""Tests for model_health_check module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_health_check import (
    weight_norm_check, activation_range_check,
    attention_health_check, prediction_quality_check,
    residual_growth_check,
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


def test_weight_norm_check_structure(model):
    result = weight_norm_check(model)
    assert result['embed_norm'] > 0
    assert result['unembed_norm'] > 0
    assert len(result['per_layer']) == 2


def test_weight_norm_check_positive(model):
    result = weight_norm_check(model)
    for p in result['per_layer']:
        assert p['W_Q_norm'] > 0
        assert p['W_K_norm'] > 0
        assert p['W_V_norm'] > 0


def test_activation_range_check_structure(model, tokens):
    result = activation_range_check(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert isinstance(p['has_large_activations'], bool)


def test_activation_range_check_values(model, tokens):
    result = activation_range_check(model, tokens)
    for p in result['per_layer']:
        assert p['resid_max'] > 0
        assert 0 <= p['mlp_sparsity'] <= 1


def test_attention_health_check_structure(model, tokens):
    result = attention_health_check(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert isinstance(h['is_degenerate'], bool)
        assert h['last_pos_entropy'] >= 0


def test_prediction_quality_check_structure(model, tokens):
    result = prediction_quality_check(model, tokens)
    assert len(result['per_position']) == 4  # seq_len - 1
    assert 0 <= result['accuracy'] <= 1


def test_prediction_quality_check_probs(model, tokens):
    result = prediction_quality_check(model, tokens)
    for p in result['per_position']:
        assert 0 <= p['top_prob'] <= 1
        assert 0 <= p['next_token_prob'] <= 1
        assert isinstance(p['correct'], bool)


def test_residual_growth_check_structure(model, tokens):
    result = residual_growth_check(model, tokens)
    assert result['embed_norm'] > 0
    assert result['final_norm'] > 0
    assert len(result['per_layer']) == 2


def test_residual_growth_check_values(model, tokens):
    result = residual_growth_check(model, tokens)
    for p in result['per_layer']:
        assert p['growth_factor'] > 0
        assert isinstance(p['is_exploding'], bool)
        assert isinstance(p['is_collapsing'], bool)
