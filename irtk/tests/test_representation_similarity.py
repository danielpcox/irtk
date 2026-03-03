"""Tests for representation_similarity module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.representation_similarity import (
    layer_representation_similarity,
    position_representation_similarity,
    component_output_similarity,
    representation_drift,
    cross_input_similarity,
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


def test_layer_representation_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_representation_similarity(model, tokens)
    n = len(result['stage_names'])
    assert result['similarity_matrix'].shape == (n, n)
    # Diagonal should be 1.0
    for i in range(n):
        assert abs(float(result['similarity_matrix'][i, i]) - 1.0) < 0.01
    assert -1.0 <= result['mean_similarity'] <= 1.0


def test_position_representation_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_representation_similarity(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['position_similarity'].shape == (5, 5)
        assert -1.0 <= p['mean_similarity'] <= 1.0


def test_component_output_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_output_similarity(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1.0 <= p['attn_mlp_cosine'] <= 1.0
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0


def test_representation_drift(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_drift(model, tokens)
    assert len(result['per_layer']) >= 2
    for p in result['per_layer']:
        assert p['l2_distance'] >= 0
        assert p['relative_change'] >= 0
    assert result['total_drift'] >= 0


def test_cross_input_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    tokens2 = jnp.array([5, 15, 25, 35, 45])
    result = cross_input_similarity(model, tokens, tokens2)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1.0 <= p['mean_cosine'] <= 1.0
        assert len(p['per_position']) == 5


def test_similarity_matrix_symmetric(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_representation_similarity(model, tokens)
    mat = result['similarity_matrix']
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            assert abs(float(mat[i, j] - mat[j, i])) < 0.01


def test_drift_has_total(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_drift(model, tokens)
    total = sum(p['l2_distance'] for p in result['per_layer'])
    assert abs(result['total_drift'] - total) < 0.01


def test_cross_input_same_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_input_similarity(model, tokens, tokens)
    for p in result['per_layer']:
        # Same input should give high similarity
        assert p['mean_cosine'] > 0.9


def test_component_alignment_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_output_similarity(model, tokens)
    for p in result['per_layer']:
        assert -1.0 <= p['alignment'] <= 1.0
