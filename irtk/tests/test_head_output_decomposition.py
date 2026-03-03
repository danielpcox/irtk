"""Tests for head_output_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_output_decomposition import (
    head_residual_contribution, head_logit_projection,
    head_value_decomposition, head_output_interference,
    head_output_rank_analysis,
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


def test_head_residual_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_residual_contribution(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    assert 'residual_norm' in result
    for h in result['per_head']:
        assert 'mean_norm' in h
        assert h['mean_norm'] >= 0
        assert isinstance(h['is_constructive'], bool)


def test_head_residual_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_residual_contribution(model, tokens, layer=1)
    norms = [h['mean_norm'] for h in result['per_head']]
    assert norms == sorted(norms, reverse=True)


def test_head_logit_projection(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_logit_projection(model, tokens, layer=0, position=-1)
    assert result['layer'] == 0
    assert result['position'] == 4
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'logit_norm' in h
        assert len(h['top_promoted']) == 5
        assert len(h['top_suppressed']) == 5


def test_head_logit_projection_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_logit_projection(model, tokens, layer=0)
    norms = [h['logit_norm'] for h in result['per_head']]
    assert norms == sorted(norms, reverse=True)


def test_head_value_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_value_decomposition(model, tokens, layer=0, head=0)
    assert result['layer'] == 0
    assert result['head'] == 0
    assert len(result['per_query']) == 5
    for q in result['per_query']:
        assert 'per_source' in q
        assert len(q['per_source']) <= q['query_position'] + 1


def test_head_output_interference(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_interference(model, tokens, layer=0, position=-1)
    assert result['layer'] == 0
    assert result['position'] == 4
    assert 'interference_ratio' in result
    assert result['interference_ratio'] >= 0
    assert isinstance(result['is_mostly_constructive'], bool)
    assert 'n_constructive_pairs' in result


def test_head_interference_pairs(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_interference(model, tokens, layer=0)
    # 4 heads → C(4,2) = 6 pairs
    assert len(result['pairs']) == 6
    for p in result['pairs']:
        assert -1.01 <= p['cosine_similarity'] <= 1.01


def test_head_output_rank_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_rank_analysis(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    assert 'mean_effective_rank' in result
    for h in result['per_head']:
        assert 'effective_rank' in h
        assert h['effective_rank'] > 0
        assert 'top_sv_fraction' in h
        assert isinstance(h['is_low_rank'], bool)


def test_head_rank_sv_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_rank_analysis(model, tokens, layer=1)
    for h in result['per_head']:
        assert 0 <= h['top_sv_fraction'] <= 1.01
