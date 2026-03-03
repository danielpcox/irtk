"""Tests for weight_matrix_probing module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_matrix_probing import (
    weight_spectrum,
    weight_alignment,
    weight_norms_profile,
    low_rank_structure,
    embed_unembed_alignment,
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


def test_weight_spectrum_attn(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_spectrum(model, layer=0, component='attn')
    assert result['component'] == 'attn'
    assert 'W_Q' in result['matrices']
    assert 'W_K' in result['matrices']
    for name, info in result['matrices'].items():
        assert info['effective_rank'] > 0
        assert info['max_singular_value'] > 0


def test_weight_spectrum_mlp(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_spectrum(model, layer=0, component='mlp')
    assert result['component'] == 'mlp'
    assert 'W_in' in result['matrices']
    assert 'W_out' in result['matrices']


def test_weight_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_alignment(model, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['qk_norm'] >= 0
        assert h['ov_norm'] >= 0
        assert h['ov_effective_rank'] > 0


def test_weight_norms_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_norms_profile(model)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['W_Q_norm'] > 0
        assert p['W_in_norm'] > 0


def test_low_rank_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = low_rank_structure(model, layer=0, rank_threshold=0.95)
    assert 'matrices' in result
    for name, info in result['matrices'].items():
        assert info['full_rank'] > 0
        assert info['rank_for_threshold'] > 0
        assert 0 < info['compression_ratio'] <= 1.0


def test_embed_unembed_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = embed_unembed_alignment(model, top_k=5)
    assert len(result['most_aligned']) == 5
    assert len(result['least_aligned']) == 5
    assert -1.0 <= result['mean_alignment'] <= 1.0


def test_low_rank_count(model_and_tokens):
    model, tokens = model_and_tokens
    result = low_rank_structure(model, layer=0)
    assert result['n_low_rank'] >= 0


def test_spectrum_variance_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_spectrum(model, layer=0, component='attn')
    for name, info in result['matrices'].items():
        assert 0 <= info['top1_variance_fraction'] <= 1.0
        assert info['top5_variance_fraction'] >= info['top1_variance_fraction']
