"""Tests for logit_gradient_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_gradient_analysis import (
    logit_sensitivity_profile,
    logit_jacobian_structure,
    per_dimension_logit_impact,
    gradient_alignment_across_layers,
    logit_curvature,
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


def test_logit_sensitivity_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_sensitivity_profile(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['sensitivity'] >= 0
        assert 0 <= p['relative_sensitivity'] <= 1.0


def test_logit_jacobian_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_jacobian_structure(model, tokens)
    assert result['effective_rank'] > 0
    assert len(result['top_singular_values']) == 5
    assert 0 <= result['concentration'] <= 1.0
    assert result['condition_number'] >= 1.0


def test_per_dimension_logit_impact(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_dimension_logit_impact(model, tokens, top_k=5)
    assert len(result['top_dimensions']) == 5
    assert result['total_impact'] >= 0
    assert 0 <= result['concentration'] <= 1.0
    for d in result['top_dimensions']:
        assert 0 <= d['dimension'] < 16  # d_model=16


def test_gradient_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = gradient_alignment_across_layers(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['gradient_norm'] >= 0
    assert -1.0 <= result['mean_alignment'] <= 1.0


def test_logit_curvature(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_curvature(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['curvature'] >= 0


def test_sensitivity_with_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_sensitivity_profile(model, tokens, target_token=5)
    assert result['target_token'] == 5
    assert len(result['per_layer']) == 2


def test_dimension_impact_sums(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_dimension_logit_impact(model, tokens, top_k=16)  # all dims
    # Top-k concentration should be ~1.0 when k=d_model
    assert result['concentration'] > 0.99


def test_jacobian_singular_values_decreasing(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_jacobian_structure(model, tokens, top_k=5)
    svs = result['top_singular_values']
    for i in range(len(svs) - 1):
        assert svs[i] >= svs[i + 1] - 0.01  # allow small float error


def test_curvature_epsilon(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_curvature(model, tokens, epsilon=0.001)
    assert len(result['per_layer']) == 2
