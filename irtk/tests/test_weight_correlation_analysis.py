"""Tests for weight_correlation_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_correlation_analysis import (
    cross_layer_weight_correlation,
    weight_norm_pattern,
    head_weight_similarity,
    qk_ov_weight_balance,
    weight_initialization_deviation,
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
    return model


def test_cross_layer_weight_correlation(model_and_tokens):
    model = model_and_tokens
    result = cross_layer_weight_correlation(model, 'W_Q')
    assert result['correlation_matrix'].shape == (2, 2)
    assert len(result['per_pair']) == 1
    assert -1.0 <= result['mean_correlation'] <= 1.0


def test_weight_norm_pattern(model_and_tokens):
    model = model_and_tokens
    result = weight_norm_pattern(model)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['total_norm'] > 0
        assert 'W_Q' in p['norms']


def test_head_weight_similarity(model_and_tokens):
    model = model_and_tokens
    result = head_weight_similarity(model, layer=0)
    assert len(result['per_pair']) == 6  # C(4,2)
    for p in result['per_pair']:
        assert -1.0 <= p['q_similarity'] <= 1.0
        assert -1.0 <= p['mean_similarity'] <= 1.0


def test_qk_ov_weight_balance(model_and_tokens):
    model = model_and_tokens
    result = qk_ov_weight_balance(model)
    assert len(result['per_layer']) == 2
    for l in result['per_layer']:
        assert len(l['per_head']) == 4
        for h in l['per_head']:
            assert 0 <= h['qk_fraction'] <= 1.0


def test_weight_initialization_deviation(model_and_tokens):
    model = model_and_tokens
    result = weight_initialization_deviation(model)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'W_Q' in p['deviations']
        assert p['deviations']['W_Q']['std'] > 0


def test_correlation_symmetric(model_and_tokens):
    model = model_and_tokens
    result = cross_layer_weight_correlation(model, 'W_K')
    mat = result['correlation_matrix']
    assert abs(float(mat[0, 1]) - float(mat[1, 0])) < 0.01


def test_different_matrix_types(model_and_tokens):
    model = model_and_tokens
    for mt in ['W_Q', 'W_K', 'W_V', 'W_O', 'W_in', 'W_out']:
        result = cross_layer_weight_correlation(model, mt)
        assert len(result['per_pair']) >= 1


def test_norm_pattern_has_all_matrices(model_and_tokens):
    model = model_and_tokens
    result = weight_norm_pattern(model)
    for p in result['per_layer']:
        for name in ['W_Q', 'W_K', 'W_V', 'W_O', 'W_in', 'W_out']:
            assert name in p['norms']


def test_deviation_kurtosis(model_and_tokens):
    model = model_and_tokens
    result = weight_initialization_deviation(model)
    # Random normal init should have kurtosis near 0
    for p in result['per_layer']:
        for name, dev in p['deviations'].items():
            assert abs(dev['kurtosis']) < 5.0  # loose bound
