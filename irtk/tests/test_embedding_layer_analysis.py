"""Tests for embedding_layer_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_layer_analysis import (
    embedding_norm_structure, embedding_similarity_to_unembed,
    embedding_neighborhood, positional_embedding_structure,
    embedding_effective_dimension,
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


def test_embedding_norm_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_norm_structure(model, tokens)
    assert result['global_mean_norm'] >= 0
    assert result['global_std_norm'] >= 0
    assert result['global_max_norm'] >= result['global_min_norm']
    assert len(result['per_token']) == 5
    for p in result['per_token']:
        assert p['norm'] >= 0
        assert 0 <= p['percentile'] <= 1
        assert isinstance(p['is_outlier'], bool)


def test_embedding_similarity_to_unembed(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_similarity_to_unembed(model, tokens)
    assert len(result['per_token']) == 5
    assert 'mean_embed_unembed_cos' in result
    for p in result['per_token']:
        assert -1.01 <= p['embed_unembed_cos'] <= 1.01
        assert isinstance(p['is_self_promoting'], bool)


def test_embedding_neighborhood(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_neighborhood(model, tokens, top_k=3)
    assert len(result['per_token']) == 5
    for p in result['per_token']:
        assert len(p['neighbors']) == 3
        for n in p['neighbors']:
            assert 'token' in n
            assert 'similarity' in n
            assert n['token'] != p['token']


def test_embedding_neighborhood_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_neighborhood(model, tokens, top_k=3)
    for p in result['per_token']:
        sims = [n['similarity'] for n in p['neighbors']]
        assert sims == sorted(sims, reverse=True)


def test_positional_embedding_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_embedding_structure(model, tokens)
    assert len(result['per_position']) == 5
    assert 'mean_position_similarity' in result
    for p in result['per_position']:
        assert 'norm' in p
        assert p['norm'] >= 0
        assert 'content_position_cos' in p


def test_embedding_effective_dimension(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_effective_dimension(model, tokens)
    assert result['effective_dimension'] > 0
    assert result['dim_for_90_pct'] >= 1
    assert result['d_model'] == 16
    assert len(result['per_component']) >= 1
    for c in result['per_component']:
        assert c['singular_value'] >= 0
        assert 0 <= c['variance_explained'] <= 1.01


def test_effective_dimension_cumulative(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_effective_dimension(model, tokens)
    cumulatives = [c['cumulative'] for c in result['per_component']]
    assert all(cumulatives[i] <= cumulatives[i+1] + 1e-5 for i in range(len(cumulatives)-1))


def test_norm_structure_global(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_norm_structure(model, tokens)
    assert result['global_max_norm'] >= result['global_mean_norm']


def test_positional_norm_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_embedding_structure(model, tokens)
    for p in result['per_position']:
        assert p['norm'] > 0
