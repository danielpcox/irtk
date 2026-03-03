"""Tests for weight_symmetry_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_symmetry_analysis import (
    qk_symmetry, ov_symmetry,
    mlp_weight_symmetry, cross_head_symmetry,
    embed_unembed_symmetry,
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


def test_qk_symmetry_structure(model):
    result = qk_symmetry(model, layer=0, head=0)
    assert 0 <= result['symmetric_fraction'] <= 1
    assert 0 <= result['antisymmetric_fraction'] <= 1
    assert isinstance(result['is_symmetric'], bool)


def test_qk_symmetry_fractions_sum(model):
    result = qk_symmetry(model, layer=0, head=0)
    assert abs(result['symmetric_fraction'] + result['antisymmetric_fraction'] - 1.0) < 0.01


def test_ov_symmetry_structure(model):
    result = ov_symmetry(model, layer=0, head=0)
    assert 0 <= result['symmetric_fraction'] <= 1
    assert isinstance(result['is_symmetric'], bool)


def test_mlp_weight_symmetry_structure(model):
    result = mlp_weight_symmetry(model, layer=0)
    assert -1 <= result['transpose_cosine'] <= 1
    assert isinstance(result['is_approximately_transpose'], bool)


def test_mlp_weight_symmetry_norms(model):
    result = mlp_weight_symmetry(model, layer=0)
    assert result['W_in_norm'] > 0
    assert result['W_out_norm'] > 0


def test_cross_head_symmetry_structure(model):
    result = cross_head_symmetry(model, layer=0)
    assert len(result['pairs']) == 6  # C(4,2)
    assert isinstance(result['heads_are_diverse'], bool)


def test_cross_head_symmetry_cosine(model):
    result = cross_head_symmetry(model, layer=0)
    for p in result['pairs']:
        assert -1 <= p['qk_similarity'] <= 1
        assert -1 <= p['ov_similarity'] <= 1


def test_embed_unembed_symmetry_structure(model):
    result = embed_unembed_symmetry(model)
    assert -1 <= result['global_cosine'] <= 1
    assert isinstance(result['is_weight_tied'], bool)


def test_embed_unembed_symmetry_norms(model):
    result = embed_unembed_symmetry(model)
    assert result['W_E_norm'] > 0
    assert result['W_U_norm'] > 0
