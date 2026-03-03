"""Tests for model_behavior_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_behavior_profiling import (
    prediction_confidence_profile,
    attention_pattern_profile,
    computation_budget_profile,
    position_difficulty_profile,
    model_summary_stats,
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


def test_prediction_confidence_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_confidence_profile(model, tokens)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 0 <= p['top_probability'] <= 1.0
        assert p['entropy'] >= 0
    assert 0 <= result['mean_confidence'] <= 1.0


def test_attention_pattern_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_profile(model, tokens)
    assert len(result['per_layer']) == 2
    for l in result['per_layer']:
        assert len(l['per_head']) == 4
        assert l['n_sparse_heads'] >= 0


def test_computation_budget_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_budget_profile(model, tokens)
    assert len(result['per_layer']) == 2
    assert result['total_computation'] >= 0
    for b in result['per_layer']:
        assert b['attn_budget'] >= 0
        assert b['mlp_budget'] >= 0
        assert 0 <= b['attn_fraction'] <= 1.0


def test_position_difficulty_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_difficulty_profile(model, tokens)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 0 <= p['difficulty'] <= 1.01
        assert 0 <= p['confidence'] <= 1.0


def test_model_summary_stats(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_summary_stats(model, tokens)
    assert result['n_layers'] == 2
    assert result['seq_len'] == 5
    assert 0 <= result['mean_top_probability'] <= 1.0
    assert result['residual_growth'] > 0
    assert result['total_heads'] == 8


def test_confidence_top5_mass(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_confidence_profile(model, tokens)
    for p in result['per_position']:
        assert p['top5_mass'] >= p['top_probability']


def test_budget_fractions_sum(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_budget_profile(model, tokens)
    total_frac = sum(b['fraction_of_total'] for b in result['per_layer'])
    assert abs(total_frac - 1.0) < 0.01


def test_difficulty_ranks_unique(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_difficulty_profile(model, tokens)
    ranks = sorted(p['difficulty_rank'] for p in result['per_position'])
    assert ranks == list(range(1, 6))


def test_attention_entropy_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_profile(model, tokens)
    for l in result['per_layer']:
        for h in l['per_head']:
            assert h['mean_entropy'] >= 0
