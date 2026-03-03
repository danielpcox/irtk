"""Tests for layer_specialization_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_specialization_profiling import (
    layer_prediction_impact,
    layer_information_added,
    layer_uniqueness,
    attn_vs_mlp_specialization,
    layer_role_classification,
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


def test_layer_prediction_impact(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_impact(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['kl_divergence'] >= 0
        assert p['logit_delta_norm'] >= 0


def test_layer_information_added(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_information_added(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_new_info'] >= 0
        assert p['mlp_new_info'] >= 0


def test_layer_uniqueness(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uniqueness(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['uniqueness'] <= 1.01
        assert p['delta_norm'] >= 0


def test_attn_vs_mlp_specialization(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_vs_mlp_specialization(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['attn_logit_fraction'] <= 1.0


def test_layer_role_classification(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_role_classification(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['role'] in ('passthrough', 'refining', 'transforming')
        assert p['relative_change'] >= 0


def test_prediction_impact_types(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_impact(model, tokens)
    for p in result['per_layer']:
        assert isinstance(p['prediction_changed'], bool)


def test_uniqueness_sums(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_uniqueness(model, tokens)
    for p in result['per_layer']:
        assert abs(p['uniqueness'] + p['mean_similarity_to_others'] - 1.0) < 0.01


def test_specialization_has_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_vs_mlp_specialization(model, tokens)
    assert 0 <= result['target_token'] < 50


def test_role_alignment_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_role_classification(model, tokens)
    for p in result['per_layer']:
        assert -1.0 <= p['pre_delta_alignment'] <= 1.0
