"""Tests for embedding_space_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_space_analysis import (
    embedding_isotropy,
    embedding_neighborhood,
    embed_unembed_correspondence,
    embedding_norm_distribution,
    embedding_subspace_structure,
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
    return model


def test_embedding_isotropy(model_and_tokens):
    model = model_and_tokens
    result = embedding_isotropy(model)
    assert result['embedding']['effective_dimensionality'] > 0
    assert result['unembedding']['effective_dimensionality'] > 0
    assert result['embedding']['isotropy_ratio'] >= 0


def test_embedding_neighborhood(model_and_tokens):
    model = model_and_tokens
    result = embedding_neighborhood(model, [0, 10, 20], k=5)
    assert len(result['per_token']) == 3
    for t in result['per_token']:
        assert len(t['neighbors']) == 5
        assert t['embedding_norm'] >= 0


def test_embed_unembed_correspondence(model_and_tokens):
    model = model_and_tokens
    result = embed_unembed_correspondence(model, top_k=5)
    assert len(result['most_aligned']) == 5
    assert len(result['least_aligned']) == 5
    assert -1.0 <= result['mean_alignment'] <= 1.0


def test_embedding_norm_distribution(model_and_tokens):
    model = model_and_tokens
    result = embedding_norm_distribution(model)
    assert result['mean_norm'] > 0
    assert result['max_norm'] >= result['min_norm']
    assert result['n_tokens'] == 50


def test_embedding_subspace_structure(model_and_tokens):
    model = model_and_tokens
    result = embedding_subspace_structure(model, n_components=5)
    assert len(result['explained_variance']) == 5
    assert result['dims_for_90pct'] > 0
    # Variance fractions should be valid
    for v in result['explained_variance']:
        assert 0 <= v <= 1.0


def test_neighborhood_self_excluded(model_and_tokens):
    model = model_and_tokens
    result = embedding_neighborhood(model, [5], k=3)
    for n in result['per_token'][0]['neighbors']:
        assert n['token'] != 5


def test_subspace_cumulative(model_and_tokens):
    model = model_and_tokens
    result = embedding_subspace_structure(model, n_components=5)
    for i in range(1, len(result['cumulative_variance'])):
        assert result['cumulative_variance'][i] >= result['cumulative_variance'][i - 1] - 0.01


def test_isotropy_condition_number(model_and_tokens):
    model = model_and_tokens
    result = embedding_isotropy(model)
    assert result['embedding']['condition_number'] >= 1.0


def test_norm_distribution_cv(model_and_tokens):
    model = model_and_tokens
    result = embedding_norm_distribution(model)
    assert result['cv'] >= 0
