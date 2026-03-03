"""Tests for token_influence_tracking module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_influence_tracking import (
    token_influence_via_attention,
    token_influence_via_residual,
    multi_token_influence,
    influence_path_analysis,
    influence_on_logits,
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


def test_token_influence_via_attention(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_influence_via_attention(model, tokens, source_pos=0)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['direct_attention'] <= 1.0
        assert len(p['per_head_attention']) == 4


def test_token_influence_via_residual(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_influence_via_residual(model, tokens, source_pos=0)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1.0 <= p['influence'] <= 1.0


def test_multi_token_influence(model_and_tokens):
    model, tokens = model_and_tokens
    result = multi_token_influence(model, tokens)
    assert len(result['per_position']) == 5
    assert 0 <= result['most_influential'] < 5
    assert result['influence_entropy'] >= 0


def test_influence_path_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = influence_path_analysis(model, tokens, source_pos=0)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert len(p['head_paths']) == 4
        for hp in p['head_paths']:
            assert hp['path_strength'] >= 0


def test_influence_on_logits(model_and_tokens):
    model, tokens = model_and_tokens
    result = influence_on_logits(model, tokens, source_pos=0, top_k=5)
    assert len(result['top_promoted']) == 5
    assert len(result['top_demoted']) == 5
    assert result['total_logit_change_norm'] >= 0


def test_attention_influence_different_sources(model_and_tokens):
    model, tokens = model_and_tokens
    r0 = token_influence_via_attention(model, tokens, source_pos=0)
    r1 = token_influence_via_attention(model, tokens, source_pos=2)
    # Different sources should give different patterns
    assert r0['source_pos'] == 0
    assert r1['source_pos'] == 2


def test_multi_token_influence_sums_to_one(model_and_tokens):
    model, tokens = model_and_tokens
    result = multi_token_influence(model, tokens)
    total = sum(p['influence'] for p in result['per_position'])
    assert abs(total - 1.0) < 0.01


def test_influence_path_positions(model_and_tokens):
    model, tokens = model_and_tokens
    result = influence_path_analysis(model, tokens, source_pos=2, target_pos=4)
    assert result['source_pos'] == 2
    assert result['target_pos'] == 4


def test_logit_influence_has_token_ids(model_and_tokens):
    model, tokens = model_and_tokens
    result = influence_on_logits(model, tokens, source_pos=0)
    for t in result['top_promoted']:
        assert 0 <= t['token'] < 50
