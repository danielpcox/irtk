"""Tests for head_specialization_scoring module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_specialization_scoring import (
    induction_head_score, previous_token_head_score,
    copy_head_score, inhibition_head_score,
    head_role_summary,
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
    return jnp.array([1, 5, 10, 5, 10])  # repeated tokens for induction


def test_induction_head_score_structure(model, tokens):
    result = induction_head_score(model, tokens)
    assert len(result['per_head']) == 8
    assert 'n_induction' in result
    for h in result['per_head']:
        assert isinstance(h['is_induction'], bool)


def test_induction_head_score_range(model, tokens):
    result = induction_head_score(model, tokens)
    for h in result['per_head']:
        assert 0 <= h['induction_score'] <= 1


def test_previous_token_head_score_structure(model, tokens):
    result = previous_token_head_score(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert isinstance(h['is_prev_token'], bool)
        assert 0 <= h['prev_token_score'] <= 1


def test_copy_head_score_structure(model, tokens):
    result = copy_head_score(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert isinstance(h['is_copy'], bool)
        assert h['mean_copy_rank'] >= 0


def test_inhibition_head_score_structure(model, tokens):
    result = inhibition_head_score(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert isinstance(h['is_inhibition'], bool)
        assert 'mean_attended_logit' in h


def test_head_role_summary_structure(model, tokens):
    result = head_role_summary(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert 'roles' in h
        assert len(h['roles']) > 0
        assert 'induction_score' in h


def test_head_role_summary_has_all_scores(model, tokens):
    result = head_role_summary(model, tokens)
    for h in result['per_head']:
        assert 'prev_token_score' in h
        assert 'copy_rank' in h
        assert 'inhibition_logit' in h


def test_induction_with_no_repeats(model):
    tokens = jnp.array([1, 2, 3, 4, 5])  # no repeats
    result = induction_head_score(model, tokens)
    for h in result['per_head']:
        assert h['n_opportunities'] == 0
        assert h['induction_score'] == 0.0


def test_all_heads_covered(model, tokens):
    result = head_role_summary(model, tokens)
    heads = {(h['layer'], h['head']) for h in result['per_head']}
    expected = {(l, h) for l in range(2) for h in range(4)}
    assert heads == expected
