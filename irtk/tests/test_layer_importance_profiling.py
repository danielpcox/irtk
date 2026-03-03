"""Tests for layer_importance_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_importance_profiling import (
    layer_ablation_importance,
    layer_gradient_importance,
    layer_output_magnitude,
    layer_prediction_impact,
    cumulative_layer_importance,
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


def test_layer_ablation_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_ablation_importance(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['kl_divergence'] >= 0
    assert 0 <= result['most_important'] < 2
    assert 0 <= result['least_important'] < 2


def test_layer_gradient_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_gradient_importance(model, tokens, position=-1)
    assert len(result['per_layer']) == 2
    assert result['position'] == 4
    for p in result['per_layer']:
        assert p['residual_norm'] >= 0
        assert 0 <= p['relative_norm'] <= 1.01


def test_layer_output_magnitude(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_output_magnitude(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0
        assert 0 <= p['attn_fraction'] <= 1.0


def test_layer_prediction_impact(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_impact(model, tokens)
    assert len(result['per_layer']) == 2
    assert isinstance(result['n_prediction_changes'], int)
    assert 0 <= result['biggest_shift_layer'] < 2


def test_cumulative_layer_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_layer_importance(model, tokens)
    assert len(result['per_layer']) == 2
    assert isinstance(result['target_token'], int)


def test_ablation_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_ablation_importance(model, tokens)
    for i in range(len(result['per_layer']) - 1):
        assert result['per_layer'][i]['kl_divergence'] >= result['per_layer'][i+1]['kl_divergence'] - 0.01


def test_output_magnitude_totals(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_output_magnitude(model, tokens)
    assert result['attn_dominated_layers'] + result['mlp_dominated_layers'] == 2


def test_cumulative_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_layer_importance(model, tokens)
    last = result['per_layer'][-1]
    assert abs(last['fraction_of_final'] - 1.0) < 0.01


def test_gradient_importance_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_gradient_importance(model, tokens)
    assert 0 <= result['target_token'] < 50
