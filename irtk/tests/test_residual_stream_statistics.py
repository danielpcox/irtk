"""Tests for residual_stream_statistics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_statistics import (
    residual_norm_profile, residual_direction_drift,
    residual_variance_decomposition, residual_position_similarity,
    residual_component_balance,
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


def test_residual_norm_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_norm_profile(model, tokens)
    assert 'embed_norm' in result
    assert result['embed_norm'] >= 0
    assert len(result['per_layer']) == 2
    assert 'final_norm' in result
    for p in result['per_layer']:
        assert 'mean_norm' in p
        assert p['mean_norm'] >= 0


def test_norm_profile_growth(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_norm_profile(model, tokens)
    assert 'total_growth' in result


def test_residual_direction_drift(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_direction_drift(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'mean_drift' in result
    for p in result['per_layer']:
        assert -1.01 <= p['cos_to_previous'] <= 1.01
        assert -1.01 <= p['cos_to_embed'] <= 1.01


def test_residual_variance_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_variance_decomposition(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['residual_variance'] >= 0
        assert p['attention_variance'] >= 0
        assert p['mlp_variance'] >= 0
        assert 0 <= p['attn_fraction'] <= 1.01
        assert 0 <= p['mlp_fraction'] <= 1.01


def test_residual_position_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_position_similarity(model, tokens, layer=0)
    assert result['layer'] == 0
    assert 'mean_similarity' in result
    assert isinstance(result['is_converged'], bool)
    # C(5,2) = 10 pairs
    assert len(result['pairs']) == 10


def test_position_similarity_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_position_similarity(model, tokens, layer=1)
    sims = [p['similarity'] for p in result['pairs']]
    assert sims == sorted(sims, reverse=True)


def test_residual_component_balance(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_component_balance(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_attn_dominant' in result
    assert 'n_mlp_dominant' in result
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0
        assert p['dominant'] in ('attn', 'mlp')
        assert isinstance(p['is_cooperative'], bool)


def test_component_balance_ratio(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_component_balance(model, tokens)
    for p in result['per_layer']:
        assert p['attn_mlp_ratio'] >= 0


def test_component_balance_counts(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_component_balance(model, tokens)
    assert result['n_attn_dominant'] + result['n_mlp_dominant'] == 2
