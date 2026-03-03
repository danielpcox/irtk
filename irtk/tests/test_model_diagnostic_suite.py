"""Tests for model_diagnostic_suite module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_diagnostic_suite import (
    model_health_check,
    computation_budget_profile,
    prediction_quality_summary,
    attention_health_summary,
    residual_stream_health,
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


def test_model_health_check(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_health_check(model, tokens)
    assert result['is_healthy'] is True
    assert result['n_issues'] == 0
    assert len(result['layer_stats']) == 2


def test_computation_budget_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_budget_profile(model, tokens)
    assert len(result['per_component']) == 4  # 2 layers * 2 (attn + mlp)
    total_frac = sum(c['fraction'] for c in result['per_component'])
    assert abs(total_frac - 1.0) < 0.01


def test_prediction_quality_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_quality_summary(model, tokens)
    assert len(result['per_position']) == 5
    assert result['mean_entropy'] >= 0
    assert 0 <= result['mean_confidence'] <= 1.0


def test_attention_health_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_health_summary(model, tokens)
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    assert result['n_healthy'] + result['n_degenerate'] == 8


def test_residual_stream_health(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_stream_health(model, tokens)
    assert len(result['per_layer']) == 2
    assert isinstance(result['is_stable'], bool)
    assert result['final_norm'] >= 0


def test_health_no_issues(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_health_check(model, tokens)
    assert len(result['issues']) == 0


def test_budget_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_budget_profile(model, tokens)
    for i in range(len(result['per_component']) - 1):
        assert result['per_component'][i]['magnitude'] >= result['per_component'][i+1]['magnitude'] - 0.01


def test_prediction_positions(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_quality_summary(model, tokens)
    assert 0 <= result['most_confident_position'] < 5
    assert 0 <= result['least_confident_position'] < 5


def test_residual_growth(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_stream_health(model, tokens)
    for p in result['per_layer']:
        assert p['growth_rate'] > 0
