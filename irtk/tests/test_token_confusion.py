"""Tests for token_confusion module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_confusion import (
    prediction_confusion_matrix,
    logit_competition,
    layer_resolved_confusion,
    systematic_errors,
    position_error_modes,
)


@pytest.fixture
def model_and_data():
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
    tokens_list = [
        jnp.array([1, 10, 20, 30, 40]),
        jnp.array([5, 15, 25, 35, 45]),
        jnp.array([2, 12, 22, 32, 42]),
        jnp.array([3, 13, 23, 33, 43]),
        jnp.array([4, 14, 24, 34, 44]),
    ]
    return model, tokens, tokens_list


def test_prediction_confusion_matrix(model_and_data):
    model, _, tokens_list = model_and_data
    # Use pos=-2 so there's a next token at pos=-1
    result = prediction_confusion_matrix(model, tokens_list, pos=-2)
    assert 'accuracy' in result
    assert 0 <= result['accuracy'] <= 1.0
    assert result['total_predictions'] > 0
    assert isinstance(result['top_confused'], list)


def test_confusion_with_top_k(model_and_data):
    model, _, tokens_list = model_and_data
    result = prediction_confusion_matrix(model, tokens_list, top_k=3)
    assert len(result['top_confused']) <= 3


def test_logit_competition(model_and_data):
    model, tokens, _ = model_and_data
    result = logit_competition(model, tokens)
    assert 'top_tokens' in result
    assert len(result['top_tokens']) > 0
    assert 'competition_score' in result
    assert 0 <= result['competition_score'] <= 1.0
    assert result['entropy'] >= 0
    # Check first token has highest logit
    assert result['top_tokens'][0]['gap_from_top'] == 0.0


def test_logit_competition_top_k(model_and_data):
    model, tokens, _ = model_and_data
    result = logit_competition(model, tokens, top_k=3)
    assert len(result['top_tokens']) == 3


def test_layer_resolved_confusion(model_and_data):
    model, tokens, _ = model_and_data
    result = layer_resolved_confusion(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) > 0
    assert 'n_transitions' in result
    assert result['final_prediction'] >= 0
    for entry in result['per_layer']:
        assert 'top_token' in entry
        assert 'top_logit' in entry


def test_layer_resolved_transitions(model_and_data):
    model, tokens, _ = model_and_data
    result = layer_resolved_confusion(model, tokens)
    assert isinstance(result['transitions'], list)
    for t in result['transitions']:
        assert 'from_token' in t
        assert 'to_token' in t


def test_systematic_errors(model_and_data):
    model, _, tokens_list = model_and_data
    result = systematic_errors(model, tokens_list, min_occurrences=1)
    assert 'per_token' in result
    assert 'mean_error_rate' in result
    for entry in result['per_token']:
        assert 'token' in entry
        assert 'error_rate' in entry
        assert 0 <= entry['error_rate'] <= 1.0


def test_position_error_modes(model_and_data):
    model, _, tokens_list = model_and_data
    result = position_error_modes(model, tokens_list)
    assert 'per_position' in result
    assert len(result['per_position']) == len(tokens_list[0]) - 1
    for entry in result['per_position']:
        assert 'accuracy' in entry
        assert 'avg_entropy' in entry
        assert 0 <= entry['accuracy'] <= 1.0


def test_position_error_empty(model_and_data):
    model, _, _ = model_and_data
    result = position_error_modes(model, [])
    assert result['n_examples'] == 0
