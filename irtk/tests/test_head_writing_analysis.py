"""Tests for head_writing_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_writing_analysis import (
    head_writing_directions, head_logit_writing,
    head_writing_consistency, head_residual_contribution,
    head_writing_rank,
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


def test_head_writing_directions_structure(model, tokens):
    result = head_writing_directions(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['output_norm'] >= 0


def test_head_writing_directions_pairs(model, tokens):
    result = head_writing_directions(model, tokens, layer=0)
    assert len(result['direction_pairs']) == 6  # C(4,2)
    for p in result['direction_pairs']:
        assert -1 <= p['cosine'] <= 1


def test_head_logit_writing_structure(model, tokens):
    result = head_logit_writing(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['top_promoted_logit'] >= h['top_suppressed_logit']


def test_head_writing_consistency_structure(model, tokens):
    result = head_writing_consistency(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert isinstance(h['is_consistent'], bool)
        assert -1 <= h['mean_direction_consistency'] <= 1


def test_head_residual_contribution_structure(model, tokens):
    result = head_residual_contribution(model, tokens)
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    for h in result['per_head']:
        assert isinstance(h['is_constructive'], bool)


def test_head_residual_contribution_norms(model, tokens):
    result = head_residual_contribution(model, tokens)
    for h in result['per_head']:
        assert h['output_norm'] >= 0


def test_head_writing_rank_structure(model, tokens):
    result = head_writing_rank(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['effective_rank'] > 0


def test_head_writing_rank_sv_fraction(model, tokens):
    result = head_writing_rank(model, tokens, layer=0)
    for h in result['per_head']:
        assert 0 <= h['top_sv_fraction'] <= 1
        assert isinstance(h['is_low_rank'], bool)


def test_head_logit_writing_range(model, tokens):
    result = head_logit_writing(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['logit_range'] >= 0
