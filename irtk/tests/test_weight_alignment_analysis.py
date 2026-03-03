"""Tests for weight_alignment_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_alignment_analysis import (
    qk_ov_alignment, cross_head_weight_alignment,
    cross_layer_weight_alignment, embed_weight_alignment,
    mlp_weight_alignment,
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


def test_qk_ov_alignment_structure(model, tokens):
    result = qk_ov_alignment(model, layer=0, head=0)
    assert 'qk_ov_cosine' in result
    assert 'qk_spectral_norm' in result
    assert 'ov_spectral_norm' in result
    assert isinstance(result['is_aligned'], bool)


def test_qk_ov_alignment_cosine_range(model, tokens):
    result = qk_ov_alignment(model, layer=0, head=0)
    assert -1.0 <= result['qk_ov_cosine'] <= 1.0


def test_cross_head_weight_alignment_structure(model, tokens):
    result = cross_head_weight_alignment(model, layer=0)
    assert 'pairs' in result
    assert len(result['pairs']) == 6  # C(4,2)
    assert 'mean_abs_similarity' in result
    assert isinstance(result['is_diverse'], bool)


def test_cross_head_weight_alignment_similarity_range(model, tokens):
    result = cross_head_weight_alignment(model, layer=0)
    for p in result['pairs']:
        assert -1.0 <= p['ov_similarity'] <= 1.0


def test_cross_layer_weight_alignment_structure(model, tokens):
    result = cross_layer_weight_alignment(model, component='ov')
    assert 'pairs' in result
    assert len(result['pairs']) == 1  # C(2,1)
    assert result['component'] == 'ov'


def test_cross_layer_weight_alignment_qk(model, tokens):
    result = cross_layer_weight_alignment(model, component='qk')
    assert result['component'] == 'qk'
    assert len(result['pairs']) == 1


def test_embed_weight_alignment_structure(model, tokens):
    result = embed_weight_alignment(model)
    assert 'mean_token_alignment' in result
    assert 'std_token_alignment' in result
    assert isinstance(result['is_tied'], bool)


def test_embed_weight_alignment_norms(model, tokens):
    result = embed_weight_alignment(model)
    assert result['embed_spectral_norm'] > 0
    assert result['unembed_spectral_norm'] > 0


def test_mlp_weight_alignment_structure(model, tokens):
    result = mlp_weight_alignment(model)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'mean_in_similarity' in p
        assert 'mean_out_similarity' in p
