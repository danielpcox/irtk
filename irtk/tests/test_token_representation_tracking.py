"""Tests for token_representation_tracking module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_representation_tracking import (
    token_identity_trajectory,
    position_representation_divergence,
    token_mixing_rate,
    representation_velocity,
    representation_convergence,
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


def test_token_identity_trajectory(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_identity_trajectory(model, tokens, position=-1)
    assert len(result['per_layer']) == 2
    assert result['position'] == 4
    assert result['original_token'] == 40
    for p in result['per_layer']:
        assert p['norm'] >= 0
        assert 0 <= p['confidence'] <= 1.0


def test_position_representation_divergence(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_representation_divergence(model, tokens)
    assert len(result['per_layer']) == 2
    assert isinstance(result['representations_diverge'], bool)


def test_token_mixing_rate(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_mixing_rate(model, tokens, position=-1)
    assert len(result['per_layer']) == 2
    assert result['position'] == 4
    assert isinstance(result['mixing_rate'], float)


def test_representation_velocity(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_velocity(model, tokens)
    assert len(result['per_layer']) == 2
    assert result['per_layer'][0]['mean_velocity'] == 0.0  # first layer has no previous
    assert result['per_layer'][1]['mean_velocity'] >= 0


def test_representation_convergence(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_convergence(model, tokens)
    assert len(result['per_layer']) == 2
    assert isinstance(result['representations_converge'], bool)
    assert isinstance(result['convergence'], float)


def test_identity_decay(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_identity_trajectory(model, tokens, position=0)
    assert isinstance(result['identity_decay'], float)


def test_divergence_trend(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_representation_divergence(model, tokens)
    assert isinstance(result['divergence_trend'], float)


def test_velocity_peak(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_velocity(model, tokens)
    assert 0 <= result['peak_layer'] < 2


def test_convergence_early_late(model_and_tokens):
    model, tokens = model_and_tokens
    result = representation_convergence(model, tokens)
    assert -1.0 <= result['early_similarity'] <= 1.01
    assert -1.0 <= result['late_similarity'] <= 1.01
