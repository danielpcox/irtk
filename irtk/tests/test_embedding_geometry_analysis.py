"""Tests for embedding_geometry_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_geometry_analysis import (
    embedding_isotropy, embedding_nearest_neighbors,
    embedding_pca_structure, embedding_cluster_structure,
    embedding_norm_distribution,
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


def test_embedding_isotropy_structure(model):
    result = embedding_isotropy(model)
    assert 0 <= result['isotropy'] <= 1
    assert result['effective_dimension'] > 0
    assert result['d_model'] == 16


def test_embedding_isotropy_top_fraction(model):
    result = embedding_isotropy(model)
    assert 0 <= result['top_eigenvalue_fraction'] <= 1
    assert result['mean_norm'] > 0


def test_embedding_nearest_neighbors_structure(model):
    result = embedding_nearest_neighbors(model, token_ids=[1, 5, 10], top_k=3)
    assert len(result['per_token']) == 3
    for t in result['per_token']:
        assert len(t['neighbors']) == 3
        assert t['norm'] > 0


def test_embedding_nearest_neighbors_cosine_range(model):
    result = embedding_nearest_neighbors(model, token_ids=[1, 5], top_k=3)
    for t in result['per_token']:
        for n in t['neighbors']:
            assert -1.0 <= n['cosine'] <= 1.0


def test_embedding_pca_structure(model):
    result = embedding_pca_structure(model, n_components=5)
    assert len(result['components']) == 5
    assert result['total_variance'] > 0


def test_embedding_pca_cumulative(model):
    result = embedding_pca_structure(model, n_components=5)
    for c in result['components']:
        assert 0 <= c['variance_fraction'] <= 1
        assert 0 <= c['cumulative_variance'] <= 1.01


def test_embedding_cluster_structure(model):
    result = embedding_cluster_structure(model, n_samples=20)
    assert isinstance(result['is_well_spread'], bool)
    assert -1 <= result['mean_pairwise_cosine'] <= 1


def test_embedding_norm_distribution_structure(model):
    result = embedding_norm_distribution(model, top_k=5)
    assert len(result['top_norm_tokens']) == 5
    assert len(result['bottom_norm_tokens']) == 5
    assert result['max_norm'] >= result['min_norm']


def test_embedding_norm_distribution_sorted(model):
    result = embedding_norm_distribution(model, top_k=5)
    norms = [t['norm'] for t in result['top_norm_tokens']]
    assert norms == sorted(norms, reverse=True)
