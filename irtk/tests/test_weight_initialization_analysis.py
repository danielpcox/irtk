"""Tests for weight_initialization_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_initialization_analysis import (
    weight_scale_profile,
    weight_distribution_stats,
    weight_norm_comparison,
    weight_sparsity_profile,
    embedding_weight_analysis,
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


def test_weight_scale_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_scale_profile(model)
    assert result['n_weight_matrices'] > 0
    for w in result['per_weight']:
        assert w['std'] >= 0
        assert w['max_abs'] >= 0


def test_weight_distribution_stats(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_distribution_stats(model, layer=0)
    assert len(result['per_weight']) == 6
    assert result['total_params'] > 0


def test_weight_norm_comparison(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_norm_comparison(model)
    assert len(result['per_layer']) == 2
    assert isinstance(result['is_balanced'], bool)
    assert result['max_layer_ratio'] >= 1.0


def test_weight_sparsity_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_sparsity_profile(model)
    assert len(result['per_layer']) == 2
    assert 0 <= result['total_sparsity'] <= 1.0


def test_embedding_weight_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_weight_analysis(model)
    assert result['embed_mean_norm'] > 0
    assert result['unembed_mean_norm'] > 0
    assert isinstance(result['is_isotropic'], bool)


def test_distribution_skewness(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_distribution_stats(model, layer=0)
    for w in result['per_weight']:
        assert isinstance(w['skewness'], float)
        assert isinstance(w['kurtosis'], float)


def test_sparsity_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    r1 = weight_sparsity_profile(model, threshold=0.001)
    r2 = weight_sparsity_profile(model, threshold=1.0)
    assert r1['total_sparsity'] <= r2['total_sparsity']


def test_norm_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_norm_comparison(model)
    for p in result['per_layer']:
        assert 0 <= p['attn_fraction'] <= 1.0


def test_scale_profile_has_embed(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_scale_profile(model)
    names = [w['name'] for w in result['per_weight']]
    assert 'W_E' in names
    assert 'W_U' in names
