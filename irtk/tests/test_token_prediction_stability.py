"""Tests for token_prediction_stability module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_prediction_stability import (
    prediction_layer_stability, prediction_position_consistency,
    prediction_token_competition, prediction_component_attribution,
    prediction_flip_sensitivity,
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


def test_prediction_layer_stability(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_layer_stability(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'final_prediction' in result
    assert 'commit_layer' in result
    assert len(result['per_layer']) == 2
    assert isinstance(result['is_early_commit'], bool)


def test_prediction_layer_stability_entries(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_layer_stability(model, tokens)
    for entry in result['per_layer']:
        assert 'prediction' in entry
        assert 'confidence' in entry
        assert 0 <= entry['confidence'] <= 1
        assert isinstance(entry['matches_final'], bool)


def test_prediction_position_consistency(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_position_consistency(model, tokens)
    assert len(result['per_position']) == 5
    assert 'mean_confidence' in result
    assert 'std_confidence' in result
    assert result['std_confidence'] >= 0
    assert isinstance(result['is_uniform'], bool)


def test_prediction_position_entries(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_position_consistency(model, tokens)
    for p in result['per_position']:
        assert 0 <= p['confidence'] <= 1
        assert p['entropy'] >= 0


def test_prediction_token_competition(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_token_competition(model, tokens, position=-1, top_k=3)
    assert result['position'] == 4
    assert len(result['per_token']) == 3
    assert 'margin' in result
    assert isinstance(result['is_decisive'], bool)


def test_prediction_token_competition_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_token_competition(model, tokens, top_k=3)
    for t_info in result['per_token']:
        assert len(t_info['per_layer']) == 2
        for entry in t_info['per_layer']:
            assert 'logit' in entry
            assert 'rank' in entry
            assert entry['rank'] >= 0


def test_prediction_component_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_component_attribution(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'target_token' in result
    assert 'components' in result
    # embed + 2*(attn+mlp) = 5
    assert len(result['components']) == 5
    assert 'top_component' in result


def test_prediction_component_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_component_attribution(model, tokens)
    abs_contribs = [abs(c['logit_contribution']) for c in result['components']]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_prediction_flip_sensitivity(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_flip_sensitivity(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'clean_prediction' in result
    assert 'clean_confidence' in result
    assert len(result['per_noise_level']) == 5
    assert isinstance(result['is_robust'], bool)
