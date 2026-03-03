"""Tests for attention_head_output_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_output_analysis import (
    head_writing_direction, head_unembed_alignment,
    head_output_diversity, head_position_specialization,
    head_combined_effect,
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


def test_head_writing_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_direction(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    assert 'n_consistent' in result
    for h in result['per_head']:
        assert 'direction_consistency' in h
        assert isinstance(h['is_consistent'], bool)


def test_head_writing_norms(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_direction(model, tokens, layer=1)
    for h in result['per_head']:
        assert h['mean_output_norm'] >= 0


def test_head_unembed_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_unembed_alignment(model, tokens, layer=0, position=-1)
    assert result['layer'] == 0
    assert result['position'] == 4
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'max_alignment' in h
        assert 'max_aligned_token' in h
        assert isinstance(h['is_token_specific'], bool)


def test_head_output_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_diversity(model, tokens, layer=0)
    assert result['layer'] == 0
    # C(4,2) = 6 pairs
    assert len(result['pairs']) == 6
    assert 'mean_abs_similarity' in result
    assert isinstance(result['is_diverse'], bool)


def test_diversity_similarity_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_diversity(model, tokens, layer=0)
    for p in result['pairs']:
        assert -1.01 <= p['similarity'] <= 1.01


def test_head_position_specialization(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_position_specialization(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    assert 'n_specialized' in result
    for h in result['per_head']:
        assert h['mean_norm'] >= 0
        assert h['std_norm'] >= 0
        assert 0 <= h['max_position'] < 5
        assert isinstance(h['is_position_specialized'], bool)


def test_head_combined_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_combined_effect(model, tokens, layer=0, position=-1)
    assert result['layer'] == 0
    assert result['position'] == 4
    assert 'combined_norm' in result
    assert result['combined_norm'] >= 0
    assert len(result['per_head']) == 4
    assert 'efficiency' in result
    assert 0 <= result['efficiency'] <= 1.01


def test_combined_constructive_destructive(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_combined_effect(model, tokens, layer=1)
    assert result['total_constructive'] >= 0
    assert result['total_destructive'] >= 0


def test_combined_per_head_sum(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_combined_effect(model, tokens, layer=0)
    for h in result['per_head']:
        assert isinstance(h['is_constructive'], bool)
        assert h['norm'] >= 0
