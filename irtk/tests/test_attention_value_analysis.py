"""Tests for attention_value_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_value_analysis import (
    value_vector_profile, value_weighted_output,
    value_rank_analysis, value_position_variation,
    value_unembed_projection,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([1, 5, 10, 15, 20])


def test_value_vector_profile_structure(model, tokens):
    result = value_vector_profile(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'mean_norm' in h
        assert 'direction_consistency' in h


def test_value_vector_profile_norms_positive(model, tokens):
    result = value_vector_profile(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['mean_norm'] > 0
        assert h['std_norm'] >= 0


def test_value_weighted_output_structure(model, tokens):
    result = value_weighted_output(model, tokens, layer=0, head=0)
    assert result['layer'] == 0
    assert result['head'] == 0
    assert 'per_source' in result
    assert result['weighted_output_norm'] > 0


def test_value_weighted_output_attention_sums(model, tokens):
    result = value_weighted_output(model, tokens, layer=0, head=0, position=-1)
    total_attn = sum(s['attention_weight'] for s in result['per_source'])
    assert abs(total_attn - 1.0) < 0.01


def test_value_rank_analysis_structure(model, tokens):
    result = value_rank_analysis(model, tokens, layer=0)
    assert 'per_head' in result
    assert 'mean_rank' in result
    for h in result['per_head']:
        assert 'effective_rank' in h
        assert isinstance(h['is_low_rank'], bool)


def test_value_rank_analysis_positive(model, tokens):
    result = value_rank_analysis(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['effective_rank'] > 0
        assert 0 <= h['top_sv_fraction'] <= 1


def test_value_position_variation_structure(model, tokens):
    result = value_position_variation(model, tokens, layer=0)
    assert 'per_head' in result
    assert 'n_position_dependent' in result
    for h in result['per_head']:
        assert 'direction_variation' in h
        assert isinstance(h['is_position_dependent'], bool)


def test_value_unembed_projection_structure(model, tokens):
    result = value_unembed_projection(model, tokens, layer=0, head=0)
    assert result['layer'] == 0
    assert result['head'] == 0
    assert len(result['per_source']) > 0


def test_value_unembed_projection_top_tokens(model, tokens):
    result = value_unembed_projection(model, tokens, layer=0, head=0, position=-1)
    for s in result['per_source']:
        assert len(s['top_tokens']) == 3
        assert len(s['bottom_tokens']) == 3
        assert s['output_norm'] > 0
