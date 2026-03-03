"""Tests for head_value_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_value_analysis import (
    value_vector_analysis,
    head_output_decomposition,
    value_weighted_attention,
    value_rank_analysis,
    head_writing_direction,
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


def test_value_vector_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_vector_analysis(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['value_norm'] >= 0
        assert h['position_variation'] >= 0
        assert len(h['value_norms']) == 5


def test_head_output_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_decomposition(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['output_norm'] >= 0
        assert -1.0 <= h['unembed_alignment'] <= 1.0


def test_value_weighted_attention(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_weighted_attention(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 0 <= h['concentration'] <= 1.0
        assert len(h['source_contributions']) == 5
        assert 0 <= h['dominant_source'] < 5


def test_value_rank_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_rank_analysis(model, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['effective_rank'] > 0
        assert h['top_singular_value'] > 0
        assert h['condition_number'] >= 1.0


def test_head_writing_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_direction(model, tokens, layer=0, top_k=3)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['output_norm'] >= 0
        assert len(h['top_token_alignments']) == 3


def test_value_analysis_layer1(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_vector_analysis(model, tokens, layer=1)
    assert result['layer'] == 1
    assert len(result['per_head']) == 4


def test_output_decomposition_has_top_token(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_decomposition(model, tokens, layer=0)
    assert 0 <= result['top_token'] < 50


def test_value_weighted_query_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_weighted_attention(model, tokens, layer=0, pos=2)
    assert result['query_position'] == 2


def test_rank_analysis_no_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    # value_rank_analysis doesn't need tokens
    result = value_rank_analysis(model, layer=0)
    assert len(result['per_head']) == 4
