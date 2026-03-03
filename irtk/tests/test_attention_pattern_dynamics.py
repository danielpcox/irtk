"""Tests for attention_pattern_dynamics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_dynamics import (
    position_dependent_attention,
    attention_shift_between_contexts,
    attention_entropy_profile,
    attention_distance_profile,
    attention_pattern_rank,
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


def test_position_dependent_attention(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_dependent_attention(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['entropy'] >= 0
        assert 0 <= p['max_attention'] <= 1.0


def test_attention_shift_between_contexts(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    result = attention_shift_between_contexts(model, tokens, tokens_b, layer=0, head=0)
    assert len(result['per_position']) == 5
    assert result['mean_js_divergence'] >= 0
    assert isinstance(result['is_context_sensitive'], bool)


def test_attention_entropy_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_entropy_profile(model, tokens)
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    assert result['n_sharp_heads'] + result['n_diffuse_heads'] == 8


def test_attention_distance_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_distance_profile(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5
    assert result['mean_attention_distance'] >= 0
    assert isinstance(result['is_local'], bool)


def test_attention_pattern_rank(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_rank(model, tokens, layer=0, head=0)
    assert result['effective_rank'] >= 1.0
    assert result['max_rank'] == 5
    assert 0 <= result['rank_utilization'] <= 1.01
    assert isinstance(result['is_low_rank'], bool)


def test_position_entropy_trend(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_dependent_attention(model, tokens, layer=0, head=0)
    assert isinstance(result['entropy_trend'], float)


def test_shift_js_range(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    result = attention_shift_between_contexts(model, tokens, tokens_b, layer=0, head=0)
    for p in result['per_position']:
        assert p['js_divergence'] >= 0


def test_entropy_profile_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_entropy_profile(model, tokens)
    for h in result['per_head']:
        assert h['mean_entropy'] >= 0
        assert h['min_entropy'] >= 0


def test_rank_singular_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_rank(model, tokens, layer=0, head=0)
    assert result['top_singular_value'] >= 0
    assert result['rank_90_pct'] >= 1
