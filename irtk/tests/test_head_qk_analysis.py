"""Tests for head_qk_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_qk_analysis import (
    qk_alignment_profile,
    positional_vs_content_attention,
    attention_selectivity,
    key_query_subspace,
    attention_pattern_type,
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


def test_qk_alignment_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_alignment_profile(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['q_norm'] >= 0
        assert h['dot_product_std'] >= 0


def test_positional_vs_content(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_vs_content_attention(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert isinstance(h['is_positional'], bool)
        assert h['content_variance'] >= 0


def test_attention_selectivity(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_selectivity(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['mean_entropy'] >= 0
        assert 0 <= h['normalized_entropy'] <= 1.01
        assert 0 <= h['mean_max_weight'] <= 1.0


def test_key_query_subspace(model_and_tokens):
    model, tokens = model_and_tokens
    result = key_query_subspace(model, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['effective_rank'] > 0
        assert h['condition_number'] >= 1.0


def test_attention_pattern_type(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_type(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    valid_types = {'diagonal', 'previous_token', 'uniform', 'sparse', 'mixed'}
    for h in result['per_head']:
        assert h['pattern_type'] in valid_types


def test_qk_subspace_no_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = key_query_subspace(model, layer=0)
    assert len(result['per_head']) == 4


def test_selectivity_gini_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_selectivity(model, tokens, layer=0)
    for h in result['per_head']:
        assert -0.5 <= h['gini_coefficient'] <= 1.01


def test_pattern_type_scores(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_type(model, tokens, layer=0)
    for h in result['per_head']:
        assert 0 <= h['diagonal_score'] <= 1.0
        assert 0 <= h['sparse_score'] <= 1.0


def test_positional_score_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_vs_content_attention(model, tokens, layer=0)
    for h in result['per_head']:
        assert -1.0 <= h['positional_score'] <= 1.0
