"""Tests for prediction_uncertainty module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_uncertainty import (
    layer_uncertainty_evolution,
    component_uncertainty_contribution,
    uncertainty_decomposition,
    position_uncertainty_ranking,
    uncertainty_source_localization,
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


def test_layer_uncertainty_evolution(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uncertainty_evolution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_entropy'] >= 0
        assert 0 <= p['mean_confidence'] <= 1.0


def test_component_uncertainty_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_uncertainty_contribution(model, tokens, position=-1)
    assert result['clean_entropy'] >= 0
    assert len(result['per_component']) == 4  # 2 layers * (attn + mlp)


def test_uncertainty_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = uncertainty_decomposition(model, tokens)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['total_entropy'] >= 0
        assert 0 <= p['top1_probability'] <= 1.0
        assert p['confidence_margin'] >= 0


def test_position_uncertainty_ranking(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_uncertainty_ranking(model, tokens)
    assert len(result['per_position']) == 5
    assert 0 <= result['most_uncertain'] < 5
    assert 0 <= result['most_certain'] < 5


def test_uncertainty_source_localization(model_and_tokens):
    model, tokens = model_and_tokens
    result = uncertainty_source_localization(model, tokens, position=-1)
    assert len(result['per_layer']) == 2
    assert result['final_entropy'] >= 0


def test_evolution_trend(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uncertainty_evolution(model, tokens)
    assert isinstance(result['entropy_trend'], float)
    assert isinstance(result['resolves_uncertainty'], bool)


def test_ranking_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_uncertainty_ranking(model, tokens)
    for i in range(len(result['per_position']) - 1):
        assert result['per_position'][i]['entropy'] >= result['per_position'][i+1]['entropy'] - 0.01


def test_decomposition_uncertain_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = uncertainty_decomposition(model, tokens)
    assert 0 <= result['uncertain_fraction'] <= 1.0


def test_source_localization_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = uncertainty_source_localization(model, tokens)
    assert 0 <= result['biggest_entropy_increase'] < 2
