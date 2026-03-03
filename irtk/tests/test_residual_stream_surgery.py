"""Tests for residual_stream_surgery module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_surgery import (
    project_out_direction,
    clamp_residual_norm,
    remove_component_contribution,
    add_steering_at_layer,
    dimension_clamping,
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


def test_project_out_direction(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = project_out_direction(model, tokens, direction, layer=0)
    assert result['max_logit_change'] >= 0
    assert result['mean_projection'] >= 0
    assert len(result['projection_magnitudes']) == 5


def test_clamp_residual_norm(model_and_tokens):
    model, tokens = model_and_tokens
    result = clamp_residual_norm(model, tokens, layer=0, max_norm=0.5)
    assert result['n_total'] == 5
    assert result['original_mean_norm'] >= 0
    assert result['max_logit_change'] >= 0


def test_remove_component_attn(model_and_tokens):
    model, tokens = model_and_tokens
    result = remove_component_contribution(model, tokens, layer=0, component='attn')
    assert result['component'] == 'attn'
    assert result['component_norm'] >= 0
    assert result['kl_divergence'] >= 0


def test_remove_component_mlp(model_and_tokens):
    model, tokens = model_and_tokens
    result = remove_component_contribution(model, tokens, layer=0, component='mlp')
    assert result['component'] == 'mlp'
    assert result['component_norm'] >= 0


def test_add_steering(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = add_steering_at_layer(model, tokens, direction, layer=0, alpha=1.0)
    assert len(result['top_promoted']) == 5
    assert len(result['top_demoted']) == 5
    assert result['max_logit_change'] >= 0


def test_dimension_clamping(model_and_tokens):
    model, tokens = model_and_tokens
    result = dimension_clamping(model, tokens, layer=0, dimensions=[0, 1, 2])
    assert result['n_dims_clamped'] == 3
    assert len(result['original_mean_abs_values']) == 3
    assert result['fraction_clamped'] == 3 / 16


def test_project_out_zero_direction(model_and_tokens):
    model, tokens = model_and_tokens
    # Zero direction should have no effect (it gets normalized to near-zero)
    direction = jnp.zeros(16)
    result = project_out_direction(model, tokens, direction, layer=0)
    assert result['mean_projection'] >= 0  # Just check it runs


def test_clamp_large_norm(model_and_tokens):
    model, tokens = model_and_tokens
    # Very large max_norm should not clamp anything
    result = clamp_residual_norm(model, tokens, layer=0, max_norm=1000.0)
    assert result['n_clamped'] == 0
    assert result['max_logit_change'] < 0.01


def test_steering_alpha_zero(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = add_steering_at_layer(model, tokens, direction, layer=0, alpha=0.0)
    assert result['max_logit_change'] < 0.01
