"""Tests for token_embedding_dynamics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_embedding_dynamics import (
    token_identity_evolution, embedding_residual_similarity,
    token_mixing_rate, token_representation_distance,
    token_prediction_trajectory,
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


def test_token_identity_evolution_structure(model, tokens):
    result = token_identity_evolution(model, tokens)
    assert 'input_token' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert isinstance(p['retains_identity'], bool)


def test_token_identity_evolution_probs(model, tokens):
    result = token_identity_evolution(model, tokens)
    for p in result['per_layer']:
        assert 0 <= p['input_token_prob'] <= 1
        assert 0 <= p['top_prob'] <= 1


def test_embedding_residual_similarity_structure(model, tokens):
    result = embedding_residual_similarity(model, tokens)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert len(p['per_layer']) == 2
        assert -1 <= p['final_similarity'] <= 1


def test_token_mixing_rate_structure(model, tokens):
    result = token_mixing_rate(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['mean_embed_similarity'] <= 1


def test_token_mixing_rate_range(model, tokens):
    result = token_mixing_rate(model, tokens)
    for p in result['per_layer']:
        assert p['min_similarity'] <= p['max_similarity']


def test_token_representation_distance_structure(model, tokens):
    result = token_representation_distance(model, tokens, layer=0)
    assert result['layer'] == 0
    expected_pairs = 5 * 4 // 2  # C(5,2)
    assert len(result['pairs']) == expected_pairs


def test_token_representation_distance_cosine_range(model, tokens):
    result = token_representation_distance(model, tokens, layer=0)
    for p in result['pairs']:
        assert -1.0 <= p['cosine'] <= 1.0


def test_token_prediction_trajectory_structure(model, tokens):
    result = token_prediction_trajectory(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_changes' in result
    for p in result['per_layer']:
        assert len(p['top_predictions']) == 5
        assert isinstance(p['top_changed'], bool)


def test_token_prediction_trajectory_probs(model, tokens):
    result = token_prediction_trajectory(model, tokens)
    for p in result['per_layer']:
        for pred in p['top_predictions']:
            assert 0 <= pred['prob'] <= 1
