"""Tests for feature_composition_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.feature_composition_analysis import (
    feature_amplification,
    feature_cancellation,
    cross_layer_feature_interaction,
    component_feature_alignment,
    feature_composition_scores,
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


def test_feature_amplification(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_amplification(model, tokens, direction)
    assert len(result['stages']) == 2
    for s in result['stages']:
        assert 'attn_contribution' in s
        assert 'mlp_contribution' in s


def test_feature_cancellation(model_and_tokens):
    model, tokens = model_and_tokens
    result = feature_cancellation(model, tokens)
    assert len(result['per_layer']) == 2
    for layer in result['per_layer']:
        assert 0 <= layer['cancellation_fraction'] <= 1.0


def test_cross_layer_interaction(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_feature_interaction(model, tokens, layer_a=0, layer_b=1)
    assert len(result['principal_angles']) > 0
    assert result['mean_angle'] >= 0
    assert 0 <= result['subspace_overlap'] <= 1.01


def test_component_feature_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = component_feature_alignment(model, tokens, layer=0, direction=direction)
    assert len(result['per_head']) == 4
    assert result['dominant_component'] in ('attn', 'mlp')


def test_feature_composition_scores(model_and_tokens):
    model, tokens = model_and_tokens
    result = feature_composition_scores(model, tokens)
    assert len(result['per_layer']) == 2
    for layer in result['per_layer']:
        assert layer['pre_norm'] >= 0
        assert layer['delta_norm'] >= 0
        assert isinstance(layer['is_reinforcing'], bool)


def test_amplification_stages(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(1), (16,))
    result = feature_amplification(model, tokens, direction)
    for s in result['stages']:
        assert s['amplification'] >= 0


def test_cancellation_ratio_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = feature_cancellation(model, tokens)
    for layer in result['per_layer']:
        assert -1.0 <= layer['cancellation_ratio'] <= 1.01


def test_alignment_head_projections(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(2), (16,))
    result = component_feature_alignment(model, tokens, layer=0, direction=direction)
    for h in result['per_head']:
        assert h['abs_projection'] >= 0


def test_composition_reinforcement(model_and_tokens):
    model, tokens = model_and_tokens
    result = feature_composition_scores(model, tokens)
    assert -1.0 <= result['mean_reinforcement'] <= 1.0
