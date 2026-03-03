"""Tests for residual_stream_direction_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_direction_analysis import (
    unembed_aligned_directions,
    maximally_active_dimensions,
    direction_contribution_tracking,
    residual_direction_diversity,
    important_direction_overlap,
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


def test_unembed_aligned_directions(model_and_tokens):
    model, tokens = model_and_tokens
    result = unembed_aligned_directions(model, tokens, layer=0, top_k=3)
    assert len(result['directions']) == 3
    for d in result['directions']:
        assert d['singular_value'] >= 0
        assert len(d['top_promoted']) == 3


def test_maximally_active_dimensions(model_and_tokens):
    model, tokens = model_and_tokens
    result = maximally_active_dimensions(model, tokens, layer=0, top_k=3)
    assert len(result['top_by_variance']) == 3
    assert result['total_variance'] > 0
    assert 0 < result['top_k_variance_share'] <= 1.0


def test_direction_contribution_tracking(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = direction_contribution_tracking(model, tokens, direction)
    assert len(result['layers']) == 2
    for layer in result['layers']:
        assert len(layer['per_head']) == 4


def test_residual_direction_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_direction_diversity(model, tokens)
    assert len(result['per_layer']) == 2
    for layer in result['per_layer']:
        assert layer['effective_dimensionality'] > 0


def test_important_direction_overlap(model_and_tokens):
    model, tokens = model_and_tokens
    result = important_direction_overlap(model, tokens)
    assert len(result['transitions']) == 1  # 2 layers -> 1 transition
    assert 0 <= result['mean_overlap'] <= 1.01


def test_unembed_variance_explained(model_and_tokens):
    model, tokens = model_and_tokens
    result = unembed_aligned_directions(model, tokens, layer=0, top_k=5)
    total_var = sum(d['variance_explained'] for d in result['directions'])
    assert total_var > 0
    assert total_var <= 1.01


def test_active_dims_top_token(model_and_tokens):
    model, tokens = model_and_tokens
    result = maximally_active_dimensions(model, tokens, layer=0, top_k=3)
    for d in result['top_by_variance']:
        assert 0 <= d['dimension'] < 16
        assert 0 <= d['top_logit_token'] < 50


def test_direction_tracking_cumulative(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(1), (16,))
    result = direction_contribution_tracking(model, tokens, direction)
    # Cumulative should be sum of totals
    manual_sum = sum(l['total_contribution'] for l in result['layers'])
    assert abs(result['cumulative'] - manual_sum) < 0.01


def test_diversity_bounds(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_direction_diversity(model, tokens)
    for layer in result['per_layer']:
        assert -1.0 <= layer['direction_diversity'] <= 2.0
