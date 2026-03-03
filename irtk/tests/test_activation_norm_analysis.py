"""Tests for activation_norm_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_norm_analysis import (
    residual_norm_profile,
    component_norm_comparison,
    norm_concentration,
    position_norm_variation,
    norm_growth_attribution,
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
    assert len(result['per_layer']) == 3  # embedding + 2 layers
    for p in result['per_layer']:
        assert p['mean_norm'] > 0
        assert p['max_norm'] >= p['min_norm']
    assert result['overall_growth'] > 0


def test_component_norm_comparison(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_norm_comparison(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0
        assert 0 <= p['attn_fraction'] <= 1.0


def test_norm_concentration(model_and_tokens):
    model, tokens = model_and_tokens
    result = norm_concentration(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['concentration'] <= 1.0
        assert 0 <= p['top5_norm_fraction'] <= 1.0


def test_position_norm_variation(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_norm_variation(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert len(p['norms']) == 5  # 5 tokens
        assert p['std'] >= 0
        assert p['cv'] >= 0


def test_norm_growth_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = norm_growth_attribution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        # Verify decomposition adds up
        expected = (p['attn_self_contribution'] + p['mlp_self_contribution'] +
                   p['pre_attn_interaction'] + p['pre_mlp_interaction'] +
                   p['attn_mlp_interaction'])
        assert abs(p['norm_change'] - expected) < 0.1


def test_residual_growth_rate(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_norm_profile(model, tokens)
    assert result['mean_growth_rate'] > 0


def test_concentration_vs_layer(model_and_tokens):
    model, tokens = model_and_tokens
    result = norm_concentration(model, tokens)
    # Just verify all layers have valid results
    for p in result['per_layer']:
        assert p['total_norm_squared'] > 0


def test_position_variation_ratio(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_norm_variation(model, tokens)
    for p in result['per_layer']:
        assert p['max_min_ratio'] >= 1.0
