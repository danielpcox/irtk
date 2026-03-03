"""Tests for token_contextual_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_contextual_analysis import (
    token_context_sensitivity, token_neighbor_influence,
    token_representation_divergence, token_contextual_embedding,
    token_unique_information,
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


def test_token_context_sensitivity_structure(model, tokens):
    result = token_context_sensitivity(model, tokens)
    assert len(result['per_layer']) == 2
    assert result['embed_norm'] > 0


def test_token_context_sensitivity_cosine(model, tokens):
    result = token_context_sensitivity(model, tokens)
    for p in result['per_layer']:
        assert -1 <= p['cosine_to_embed'] <= 1
        assert 0 <= p['context_influence'] <= 2


def test_token_neighbor_influence_structure(model, tokens):
    result = token_neighbor_influence(model, tokens)
    assert len(result['per_source']) > 0
    for s in result['per_source']:
        assert s['total_attention'] >= 0


def test_token_neighbor_influence_sorted(model, tokens):
    result = token_neighbor_influence(model, tokens)
    attn = [s['total_attention'] for s in result['per_source']]
    assert attn == sorted(attn, reverse=True)


def test_token_representation_divergence_structure(model, tokens):
    result = token_representation_divergence(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['mean_pairwise_cosine'] <= 1


def test_token_contextual_embedding_structure(model, tokens):
    result = token_contextual_embedding(model, tokens, layer=-1)
    assert len(result['per_token']) == 5
    for t in result['per_token']:
        assert -1 <= t['cosine_to_embed'] <= 1


def test_token_contextual_embedding_shift(model, tokens):
    result = token_contextual_embedding(model, tokens, layer=-1)
    for t in result['per_token']:
        assert 0 <= t['context_shift'] <= 2
        assert t['contextual_norm'] > 0


def test_token_unique_information_structure(model, tokens):
    result = token_unique_information(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['nearest_cosine'] <= 1
        assert p['uniqueness'] >= 0


def test_token_unique_information_position(model, tokens):
    result = token_unique_information(model, tokens, position=2)
    assert result['position'] == 2
