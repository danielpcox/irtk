"""Tests for residual_stream_attribution module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_attribution import (
    per_position_attribution,
    directional_attribution,
    component_overlap_matrix,
    cumulative_buildup,
    logit_attribution_by_component,
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


def test_per_position_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_position_attribution(model, tokens, position=-1)
    assert result['position'] == 4
    # embed + 2 layers * (attn + mlp) = 5
    assert len(result['per_component']) == 5
    assert result['total_residual_norm'] >= 0


def test_directional_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = directional_attribution(model, tokens, direction, position=-1)
    assert len(result['per_component']) == 5


def test_component_overlap_matrix(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_overlap_matrix(model, tokens, position=-1)
    assert result['n_components'] == 5
    assert result['overlap_matrix'].shape == (5, 5)
    assert result['mean_overlap'] >= 0


def test_cumulative_buildup(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_buildup(model, tokens, position=-1)
    assert len(result['stages']) == 2
    for s in result['stages']:
        assert s['residual_norm'] >= 0
        assert 0 <= s['confidence'] <= 1.0


def test_logit_attribution_by_component(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_attribution_by_component(model, tokens, position=-1)
    assert len(result['per_component']) == 5
    assert 0 <= result['target_token'] < 50


def test_attribution_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_position_attribution(model, tokens)
    for i in range(len(result['per_component']) - 1):
        assert result['per_component'][i]['norm'] >= result['per_component'][i+1]['norm'] - 0.01


def test_logit_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_attribution_by_component(model, tokens)
    for i in range(len(result['per_component']) - 1):
        assert result['per_component'][i]['abs_contribution'] >= result['per_component'][i+1]['abs_contribution'] - 0.01


def test_overlap_diagonal(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_overlap_matrix(model, tokens)
    for i in range(result['n_components']):
        assert abs(float(result['overlap_matrix'][i, i]) - 1.0) < 0.01


def test_buildup_growth(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_buildup(model, tokens)
    assert isinstance(result['norm_growth_rate'], float)
