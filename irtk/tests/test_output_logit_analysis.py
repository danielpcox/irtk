"""Tests for output_logit_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.output_logit_analysis import (
    logit_distribution_profile, logit_temperature_sensitivity,
    logit_margin_analysis, logit_rank_distribution,
    cross_position_logit_consistency,
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


def test_logit_distribution_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_distribution_profile(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'mean_logit' in result
    assert 'std_logit' in result
    assert result['std_logit'] >= 0
    assert 'skewness' in result
    assert 'entropy' in result
    assert result['entropy'] >= 0
    assert len(result['top_tokens']) == 5
    assert len(result['bottom_tokens']) == 5


def test_logit_distribution_profile_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_distribution_profile(model, tokens, position=2)
    assert result['position'] == 2
    for t in result['top_tokens']:
        assert 'token' in t
        assert 'logit' in t
        assert 'probability' in t
        assert t['probability'] >= 0


def test_logit_temperature_sensitivity(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_temperature_sensitivity(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'base_prediction' in result
    assert 'per_temperature' in result
    assert len(result['per_temperature']) == 5
    assert 'n_stable_temperatures' in result
    assert isinstance(result['is_robust'], bool)


def test_logit_temperature_entries(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_temperature_sensitivity(model, tokens)
    for entry in result['per_temperature']:
        assert 'temperature' in entry
        assert 'top_token' in entry
        assert 'confidence' in entry
        assert entry['confidence'] >= 0 and entry['confidence'] <= 1
        assert 'entropy' in entry
        assert isinstance(entry['same_prediction'], bool)


def test_logit_margin_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_margin_analysis(model, tokens)
    assert 'per_position' in result
    assert len(result['per_position']) == 5
    assert 'mean_logit_margin' in result
    assert 'n_decisive' in result
    for p in result['per_position']:
        assert 'logit_margin' in p
        assert p['logit_margin'] >= 0
        assert 'probability_margin' in p
        assert isinstance(p['is_decisive'], bool)


def test_logit_margin_top_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_margin_analysis(model, tokens)
    for p in result['per_position']:
        assert p['top1_logit'] >= p['top2_logit']


def test_logit_rank_distribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_rank_distribution(model, tokens, position=-1)
    assert result['position'] == 4
    assert 'top_1_probability' in result
    assert result['top_1_probability'] >= 0
    assert result['top_5_probability'] >= result['top_1_probability']
    assert result['top_10_probability'] >= result['top_5_probability']
    assert result['tokens_for_50_pct'] >= 1
    assert result['tokens_for_90_pct'] >= result['tokens_for_50_pct']
    assert isinstance(result['is_concentrated'], bool)


def test_logit_rank_distribution_cumulative(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_rank_distribution(model, tokens)
    assert result['top_10_probability'] <= 1.0 + 1e-5


def test_cross_position_logit_consistency(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_position_logit_consistency(model, tokens)
    assert 'per_position' in result
    assert len(result['per_position']) == 5
    assert 'mean_pairwise_kl' in result
    assert result['mean_pairwise_kl'] >= 0
    assert isinstance(result['is_consistent'], bool)
    assert 'n_outliers' in result
    for p in result['per_position']:
        assert 'mean_kl_to_others' in p
        assert p['mean_kl_to_others'] >= 0
