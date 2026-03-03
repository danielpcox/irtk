"""Tests for token_rank_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_rank_analysis import (
    token_rank_trajectory,
    rank_stability,
    rank_entropy,
    competing_tokens,
    rank_change_attribution,
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


def test_token_rank_trajectory(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_rank_trajectory(model, tokens)
    assert len(result['tracked_tokens']) == 5
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert len(p['ranks']) == 5


def test_token_rank_specific_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_rank_trajectory(model, tokens, track_tokens=[0, 1, 2])
    assert result['tracked_tokens'] == [0, 1, 2]


def test_rank_stability(model_and_tokens):
    model, tokens = model_and_tokens
    result = rank_stability(model, tokens, top_k=5)
    assert len(result['transitions']) == 1  # 2 layers -> 1 transition
    assert 0 <= result['mean_stability'] <= 1.0


def test_rank_entropy(model_and_tokens):
    model, tokens = model_and_tokens
    result = rank_entropy(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['entropy'] >= 0
        assert 0 <= p['normalized_entropy'] <= 1.0
        assert 0 <= p['max_probability'] <= 1.0


def test_competing_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = competing_tokens(model, tokens, margin=2.0)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['n_competitors'] >= 1  # At least the top token itself


def test_competing_margin(model_and_tokens):
    model, tokens = model_and_tokens
    narrow = competing_tokens(model, tokens, margin=0.1)
    wide = competing_tokens(model, tokens, margin=10.0)
    # Wider margin should have more competitors
    for n, w in zip(narrow['per_layer'], wide['per_layer']):
        assert w['n_competitors'] >= n['n_competitors']


def test_rank_change_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = rank_change_attribution(model, tokens)
    assert 'target_token' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert abs(p['total_change'] - (p['attn_logit_change'] + p['mlp_logit_change'])) < 0.01
        assert p['main_driver'] in ['attn', 'mlp']


def test_rank_change_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = rank_change_attribution(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_entropy_reduction(model_and_tokens):
    model, tokens = model_and_tokens
    result = rank_entropy(model, tokens)
    # Entropy reduction should be defined
    assert isinstance(result['entropy_reduction'], float)
