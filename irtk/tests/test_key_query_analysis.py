"""Tests for key_query_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.key_query_analysis import (
    key_query_alignment, key_query_subspace,
    key_query_position_dependence, key_query_match_profile,
    key_query_weight_decomposition,
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


def test_key_query_alignment_structure(model, tokens):
    result = key_query_alignment(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'mean_self_alignment' in h
        assert 'mean_cross_alignment' in h


def test_key_query_alignment_norms_positive(model, tokens):
    result = key_query_alignment(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['q_mean_norm'] > 0
        assert h['k_mean_norm'] > 0


def test_key_query_subspace_structure(model, tokens):
    result = key_query_subspace(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'q_effective_rank' in h
        assert 'k_effective_rank' in h


def test_key_query_subspace_ranks_positive(model, tokens):
    result = key_query_subspace(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['q_effective_rank'] > 0
        assert h['k_effective_rank'] > 0
        assert 0 <= h['q_top_sv_fraction'] <= 1
        assert 0 <= h['k_top_sv_fraction'] <= 1


def test_key_query_position_dependence_structure(model, tokens):
    result = key_query_position_dependence(model, tokens, layer=0)
    assert 'n_position_dependent' in result
    for h in result['per_head']:
        assert 'q_direction_variation' in h
        assert isinstance(h['is_position_dependent'], bool)


def test_key_query_match_profile_structure(model, tokens):
    result = key_query_match_profile(model, tokens, layer=0, head=0)
    assert result['layer'] == 0
    assert result['head'] == 0
    assert 'per_key' in result
    assert result['query_norm'] > 0


def test_key_query_match_profile_sorted(model, tokens):
    result = key_query_match_profile(model, tokens, layer=0, head=0, position=-1)
    scores = [k['score'] for k in result['per_key']]
    assert scores == sorted(scores, reverse=True)


def test_key_query_weight_decomposition_structure(model, tokens):
    result = key_query_weight_decomposition(model, layer=0, head=0)
    assert 'spectral_norm' in result
    assert 'effective_rank' in result
    assert 'symmetry_fraction' in result


def test_key_query_weight_decomposition_values(model, tokens):
    result = key_query_weight_decomposition(model, layer=0, head=0)
    assert result['spectral_norm'] > 0
    assert result['effective_rank'] > 0
    assert 0 <= result['symmetry_fraction'] <= 1
    assert 0 <= result['top_sv_fraction'] <= 1
