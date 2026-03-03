"""Tests for position_encoding_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.position_encoding_analysis import (
    position_embedding_structure,
    position_content_separation,
    position_info_persistence,
    position_attention_pattern,
    position_encoding_capacity,
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


def test_position_embedding_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_embedding_structure(model)
    assert result['has_position_embeddings'] is True
    assert result['n_positions'] == 32
    assert result['d_model'] == 16
    assert result['effective_rank'] > 0
    assert result['mean_norm'] > 0


def test_position_content_separation(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_content_separation(model, tokens)
    assert result['has_position_embeddings'] is True
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['position_fraction'] >= 0


def test_position_info_persistence(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_info_persistence(model, tokens)
    assert result['has_position_embeddings'] is True
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1.0 <= p['mean_position_cosine'] <= 1.0


def test_position_attention_pattern(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_attention_pattern(model, tokens)
    assert 'per_head' in result
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    for h in result['per_head']:
        assert 0 <= h['positional_bias'] <= 1.0
        assert 0 in h['relative_profile']  # Distance 0 always exists


def test_position_attention_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_attention_pattern(model, tokens, layers=[1])
    assert all(h['layer'] == 1 for h in result['per_head'])


def test_position_encoding_capacity(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_encoding_capacity(model)
    assert result['has_position_embeddings'] is True
    assert result['dims_for_90pct'] > 0
    assert result['dims_for_90pct'] <= result['d_model']
    assert 0 <= result['capacity_fraction_90'] <= 1.0


def test_position_encoding_capacity_custom_n(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_encoding_capacity(model, n_positions=10)
    assert result['n_positions'] == 10


def test_content_separation_total_variance(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_content_separation(model, tokens)
    for p in result['per_layer']:
        assert p['total_variance'] > 0


def test_capacity_pos_to_token_ratio(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_encoding_capacity(model)
    assert result['pos_to_token_ratio'] >= 0
    assert result['mean_position_norm'] > 0
    assert result['mean_token_norm'] > 0
