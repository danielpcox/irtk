"""Tests for prediction_formation_tracking module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_formation_tracking import (
    prediction_timeline,
    commitment_point,
    prediction_drivers,
    alternative_prediction_analysis,
    prediction_stability,
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


def test_prediction_timeline(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_timeline(model, tokens, top_k=3)
    assert len(result['stages']) == 3  # embed + 2 layers
    for s in result['stages']:
        assert len(s['top_predictions']) == 3


def test_commitment_point(model_and_tokens):
    model, tokens = model_and_tokens
    result = commitment_point(model, tokens)
    assert 0 <= result['final_token'] < 50
    assert 0 <= result['commit_layer'] <= 2
    assert len(result['trajectory']) == 2


def test_prediction_drivers(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_drivers(model, tokens)
    assert len(result['drivers']) == 5  # embed + 2*(attn+mlp)
    # Should be sorted by abs_logit
    for i in range(len(result['drivers']) - 1):
        assert result['drivers'][i]['abs_logit'] >= result['drivers'][i+1]['abs_logit']


def test_alternative_prediction_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = alternative_prediction_analysis(model, tokens, top_k=3)
    assert len(result['per_token']) == 3
    for t in result['per_token']:
        assert len(t['trajectory']) == 2


def test_prediction_stability(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_stability(model, tokens)
    assert len(result['predictions']) == 2
    assert 0 <= result['stability'] <= 1.0
    assert result['longest_streak'] >= 1


def test_timeline_embed_stage(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_timeline(model, tokens)
    assert result['stages'][0]['stage'] == 'embed'


def test_drivers_with_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_drivers(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_alternative_ranks(model_and_tokens):
    model, tokens = model_and_tokens
    result = alternative_prediction_analysis(model, tokens, top_k=2)
    for t in result['per_token']:
        for step in t['trajectory']:
            assert step['rank'] >= 0


def test_stability_changes(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_stability(model, tokens)
    assert result['n_changes'] >= 0
    assert result['n_changes'] <= len(result['predictions']) - 1
