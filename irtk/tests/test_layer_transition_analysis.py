"""Tests for layer_transition_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_transition_analysis import (
    layer_transition_magnitude,
    component_transition_contribution,
    transition_smoothness,
    identity_layers,
    critical_transitions,
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


def test_layer_transition_magnitude(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_transition_magnitude(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['delta_norm'] >= 0
        assert p['relative_change'] >= 0
        assert -1.0 <= p['direction_preservation'] <= 1.0


def test_component_transition_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_transition_contribution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert abs(p['attn_fraction'] + p['mlp_fraction'] - 1.0) < 0.01
        assert -1.0 <= p['attn_mlp_cosine'] <= 1.0


def test_transition_smoothness(model_and_tokens):
    model, tokens = model_and_tokens
    result = transition_smoothness(model, tokens)
    assert 'transitions' in result
    assert len(result['transitions']) == 1  # 2 layers -> 1 transition
    assert result['n_smooth'] + result['n_abrupt'] == 1
    assert 0 <= result['smoothness_score'] <= 1.0


def test_identity_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = identity_layers(model, tokens, threshold=0.1)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_relative_change'] >= 0
    assert result['n_identity'] == len(result['identity_layers'])


def test_identity_layers_high_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    low = identity_layers(model, tokens, threshold=0.01)
    high = identity_layers(model, tokens, threshold=100.0)
    assert high['n_identity'] >= low['n_identity']


def test_critical_transitions(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_transitions(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert abs(p['logit_delta'] - (p['attn_logit_delta'] + p['mlp_logit_delta'])) < 0.01
    assert 0 <= result['most_critical_layer'] < 2


def test_critical_transitions_target_token(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_transitions(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_transition_magnitude_pos(model_and_tokens):
    model, tokens = model_and_tokens
    r0 = layer_transition_magnitude(model, tokens, pos=0)
    r_last = layer_transition_magnitude(model, tokens, pos=-1)
    # Both should have valid results
    assert len(r0['per_layer']) == 2
    assert len(r_last['per_layer']) == 2


def test_component_contribution_cooperation(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_transition_contribution(model, tokens)
    for p in result['per_layer']:
        assert isinstance(p['cooperative'], bool)
