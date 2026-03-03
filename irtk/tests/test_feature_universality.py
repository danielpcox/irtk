"""Tests for feature_universality module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.feature_universality import (
    feature_activation_consistency,
    position_independence,
    context_invariance,
    feature_clustering_across_inputs,
    layer_wise_feature_universality,
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
    tokens_list = [
        jnp.array([1, 10, 20, 30, 40]),
        jnp.array([5, 15, 25, 35, 45]),
        jnp.array([2, 12, 22, 32, 42]),
    ]
    return model, tokens_list


def test_feature_activation_consistency(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_activation_consistency(model, tokens_list, layer=0, direction=direction)
    assert len(result['per_input']) == 3
    assert 0 <= result['consistency_score'] <= 1.0


def test_position_independence(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = position_independence(model, tokens_list[0], layer=0, direction=direction)
    assert -1.0 <= result['position_correlation'] <= 1.0
    assert isinstance(result['is_position_independent'], bool)
    assert len(result['per_position']) == 5


def test_context_invariance(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = context_invariance(model, tokens_list[0], tokens_list[1], layer=0, direction=direction)
    assert result['absolute_difference'] >= 0
    assert result['relative_difference'] >= 0
    assert isinstance(result['is_invariant'], bool)


def test_feature_clustering_across_inputs(model_and_tokens):
    model, tokens_list = model_and_tokens
    result = feature_clustering_across_inputs(model, tokens_list, layer=0, n_directions=3)
    assert len(result['per_direction']) == 3
    assert result['n_inputs'] == 3
    for d in result['per_direction']:
        assert 0 <= d['variance_explained'] <= 1.01


def test_layer_wise_feature_universality(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = layer_wise_feature_universality(model, tokens_list, direction=direction)
    assert len(result['per_layer']) == 2
    assert 0 <= result['universal_fraction'] <= 1.0


def test_consistency_single_input(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_activation_consistency(model, [tokens_list[0]], layer=0, direction=direction)
    assert len(result['per_input']) == 1


def test_same_context_invariance(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = context_invariance(model, tokens_list[0], tokens_list[0], layer=0, direction=direction)
    assert result['absolute_difference'] < 0.01


def test_clustering_variance(model_and_tokens):
    model, tokens_list = model_and_tokens
    result = feature_clustering_across_inputs(model, tokens_list, layer=0, n_directions=5)
    total_explained = sum(d['variance_explained'] for d in result['per_direction'])
    assert total_explained <= 1.01


def test_universality_per_layer(model_and_tokens):
    model, tokens_list = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = layer_wise_feature_universality(model, tokens_list, direction=direction)
    for p in result['per_layer']:
        assert p['coefficient_of_variation'] >= 0
