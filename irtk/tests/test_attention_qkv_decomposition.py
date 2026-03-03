"""Tests for attention_qkv_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_qkv_decomposition import (
    qk_eigenspectrum,
    ov_eigenspectrum,
    positional_vs_content_qk,
    value_composition_profile,
    cross_head_qkv_alignment,
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


def test_qk_eigenspectrum(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_eigenspectrum(model, layer=0, head=0)
    assert result['effective_rank'] > 0
    assert result['top_eigenvalue'] >= 0
    assert result['rank_90'] > 0


def test_ov_eigenspectrum(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_eigenspectrum(model, layer=0, head=0)
    assert result['effective_rank'] > 0
    assert result['top_singular_value'] >= 0
    assert result['rank_90'] > 0


def test_positional_vs_content_qk(model_and_tokens):
    model, tokens = model_and_tokens
    result = positional_vs_content_qk(model, tokens, layer=0, head=0)
    assert -1.0 <= result['positional_correlation'] <= 1.0
    assert isinstance(result['is_positional'], bool)


def test_value_composition_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_composition_profile(model, tokens, layer=0, head=0)
    assert result['mean_output_norm'] >= 0
    assert result['mean_value_norm'] >= 0


def test_cross_head_qkv_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_head_qkv_alignment(model, layer=0)
    assert len(result['q_alignments']) == 6  # C(4,2)
    assert len(result['ov_alignments']) == 6
    for a in result['q_alignments']:
        assert -1.0 <= a['cosine'] <= 1.01


def test_qk_all_heads(model_and_tokens):
    model, tokens = model_and_tokens
    for h in range(4):
        result = qk_eigenspectrum(model, layer=0, head=h)
        assert result['head'] == h
        assert result['effective_rank'] > 0


def test_ov_all_heads(model_and_tokens):
    model, tokens = model_and_tokens
    for h in range(4):
        result = ov_eigenspectrum(model, layer=0, head=h)
        assert result['head'] == h


def test_value_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_composition_profile(model, tokens, layer=0, head=0)
    assert result['value_diversity'] >= -1.0


def test_alignment_symmetric(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_head_qkv_alignment(model, layer=0)
    assert -1.0 <= result['mean_q_alignment'] <= 1.01
    assert -1.0 <= result['mean_ov_alignment'] <= 1.01
