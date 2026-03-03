"""Tests for distribution_shift_diagnosis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.distribution_shift_diagnosis import (
    activation_divergence_profile,
    component_vulnerability,
    feature_stability,
    layer_adaptation_difficulty,
    prediction_robustness,
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
    tokens_a = jnp.array([1, 10, 20, 30, 40])
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    return model, tokens_a, tokens_b


def test_activation_divergence_profile(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = activation_divergence_profile(model, tokens_a, tokens_b)
    assert len(result['per_hook']) > 0
    assert result['most_divergent'] is not None
    assert result['mean_divergence'] >= 0


def test_component_vulnerability(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = component_vulnerability(model, tokens_a, tokens_b)
    assert len(result['per_component']) > 0
    assert result['most_vulnerable'] is not None
    for c in result['per_component']:
        assert c['absolute_change'] >= 0
        assert c['relative_change'] >= 0


def test_feature_stability(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_stability(model, tokens_a, tokens_b, direction)
    assert len(result['per_layer']) == 2
    assert 0 <= result['stability_fraction'] <= 1.0


def test_layer_adaptation_difficulty(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = layer_adaptation_difficulty(model, tokens_a, tokens_b)
    assert len(result['per_layer']) == 2
    assert 0 <= result['hardest_layer'] < 2
    for layer in result['per_layer']:
        assert layer['l2_distance'] >= 0
        assert layer['difficulty'] >= 0


def test_prediction_robustness(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = prediction_robustness(model, tokens_a, tokens_b)
    assert len(result['per_position']) == 5
    assert 0 <= result['agreement_rate'] <= 1.0
    assert result['mean_kl_divergence'] >= 0


def test_divergence_sorted(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = activation_divergence_profile(model, tokens_a, tokens_b)
    for i in range(len(result['per_hook']) - 1):
        assert result['per_hook'][i]['l2_distance'] >= result['per_hook'][i+1]['l2_distance'] - 0.01


def test_vulnerability_sorted(model_and_tokens):
    model, tokens_a, tokens_b = model_and_tokens
    result = component_vulnerability(model, tokens_a, tokens_b)
    for i in range(len(result['per_component']) - 1):
        assert result['per_component'][i]['relative_change'] >= result['per_component'][i+1]['relative_change'] - 0.01


def test_same_input_divergence(model_and_tokens):
    model, tokens_a, _ = model_and_tokens
    result = activation_divergence_profile(model, tokens_a, tokens_a)
    for h in result['per_hook']:
        assert h['l2_distance'] < 0.01


def test_same_input_agreement(model_and_tokens):
    model, tokens_a, _ = model_and_tokens
    result = prediction_robustness(model, tokens_a, tokens_a)
    assert result['agreement_rate'] == 1.0
