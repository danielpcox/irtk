"""Tests for weight_space_geometry module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_space_geometry import (
    weight_manifold_dimension,
    weight_distance_profile,
    weight_symmetry_analysis,
    weight_interpolation_effect,
    parameter_norm_geometry,
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


def test_weight_manifold_dimension(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_manifold_dimension(model, layer=0)
    assert 'W_Q' in result['matrices']
    assert 'W_in' in result['matrices']
    for name, m in result['matrices'].items():
        assert m['effective_dimension'] > 0
        assert m['rank_utilization'] > 0


def test_weight_distance_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_distance_profile(model)
    assert result['n_layers'] == 2
    assert 'W_Q' in result['distances']
    for name, d in result['distances'].items():
        assert len(d['distance_matrix']) == 2


def test_weight_symmetry_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_symmetry_analysis(model, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert -1.0 <= h['qk_cosine'] <= 1.0
        assert -1.0 <= h['kv_cosine'] <= 1.0


def test_weight_interpolation_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_interpolation_effect(model, tokens, layer=0, matrix_name='W_in', n_steps=3)
    assert len(result['interpolation']) == 3
    # First step (alpha=0) should have no effect
    assert result['interpolation'][0]['max_logit_change'] < 0.01


def test_parameter_norm_geometry(model_and_tokens):
    model, tokens = model_and_tokens
    result = parameter_norm_geometry(model)
    assert len(result['per_layer']) == 2
    assert result['global_mean_norm'] > 0
    assert result['embed_norm'] > 0
    assert result['unembed_norm'] > 0


def test_manifold_condition_number(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_manifold_dimension(model, layer=0)
    for name, m in result['matrices'].items():
        assert m['condition_number'] >= 1.0


def test_symmetry_global(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_symmetry_analysis(model, layer=0)
    assert -1.0 <= result['global_qk_cosine'] <= 1.0
    assert -1.0 <= result['mean_qk_symmetry'] <= 1.0


def test_interpolation_attn_weight(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_interpolation_effect(model, tokens, layer=0, matrix_name='W_Q', n_steps=3)
    assert len(result['interpolation']) == 3
    assert result['matrix'] == 'W_Q'


def test_distance_symmetry(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_distance_profile(model)
    for name, d in result['distances'].items():
        # Distance matrix should be symmetric
        for i in range(2):
            for j in range(2):
                assert abs(d['distance_matrix'][i][j] - d['distance_matrix'][j][i]) < 1e-5
