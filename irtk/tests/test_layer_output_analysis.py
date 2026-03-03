"""Tests for layer_output_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_output_analysis import (
    layer_output_decomposition, layer_prediction_change,
    layer_residual_growth, layer_information_content,
    layer_uniqueness_score,
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


def test_layer_output_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_output_decomposition(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_cooperative' in result
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0
        assert 0 <= p['efficiency'] <= 1.01
        assert isinstance(p['is_cooperative'], bool)


def test_layer_output_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_output_decomposition(model, tokens)
    for p in result['per_layer']:
        assert -1.01 <= p['alignment'] <= 1.01


def test_layer_prediction_change(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_change(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'target_token' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'target_logit' in p
        assert p['target_rank'] >= 0
        assert isinstance(p['matches_final'], bool)


def test_layer_residual_growth(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_residual_growth(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['prev_norm'] >= 0
        assert p['curr_norm'] >= 0
        assert p['delta_norm'] >= 0


def test_layer_information_content(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_content(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'entropy_reduction' in result
    assert isinstance(result['sharpens_prediction'], bool)
    for p in result['per_layer']:
        assert p['mean_entropy'] >= 0
        assert 0 <= p['mean_confidence'] <= 1


def test_layer_uniqueness_score(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uniqueness_score(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_unique' in result
    for p in result['per_layer']:
        assert 0 <= p['mean_abs_similarity'] <= 1.01
        assert 0 <= p['uniqueness_score'] <= 1.01
        assert isinstance(p['is_unique'], bool)


def test_layer_uniqueness_symmetry(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uniqueness_score(model, tokens)
    # With 2 layers, each has only 1 other to compare to
    for p in result['per_layer']:
        assert p['mean_abs_similarity'] == p['max_abs_similarity']


def test_information_content_confidence(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_content(model, tokens)
    for p in result['per_layer']:
        assert p['mean_confidence'] > 0


def test_prediction_change_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_change(model, tokens, position=2)
    assert result['position'] == 2
