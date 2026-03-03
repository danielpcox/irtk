"""Tests for residual_bottleneck module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_bottleneck import (
    dimension_utilization,
    compression_points,
    redundancy_detection,
    capacity_allocation,
    critical_dimensions,
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


def test_dimension_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = dimension_utilization(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['utilization'] <= 1.0
        assert p['active_dimensions'] <= p['total_dimensions']


def test_dimension_utilization_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = dimension_utilization(model, tokens, layers=[0])
    assert len(result['per_layer']) == 1


def test_compression_points(model_and_tokens):
    model, tokens = model_and_tokens
    result = compression_points(model, tokens)
    assert 'per_layer' in result
    assert 'bottleneck_layers' in result
    for p in result['per_layer']:
        assert p['effective_rank'] > 0
        assert 'compression_ratio' in p


def test_redundancy_detection(model_and_tokens):
    model, tokens = model_and_tokens
    result = redundancy_detection(model, tokens)
    assert 'per_layer' in result
    for p in result['per_layer']:
        assert p['n_redundant_pairs'] >= 0
        assert 0 <= p['max_correlation'] <= 1.0


def test_capacity_allocation(model_and_tokens):
    model, tokens = model_and_tokens
    result = capacity_allocation(model, tokens)
    assert 'components' in result
    n_expected = 1 + 2 * model.cfg.n_layers
    assert len(result['components']) == n_expected
    for c in result['components']:
        assert c['norm_squared'] >= 0
        assert c['fraction'] >= 0


def test_capacity_allocation_pos(model_and_tokens):
    model, tokens = model_and_tokens
    result = capacity_allocation(model, tokens, pos=0)
    assert result['total_norm_squared'] > 0


def test_critical_dimensions(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_dimensions(model, tokens, top_k=3)
    assert 'per_dimension' in result
    assert len(result['per_dimension']) == 3
    assert result['target_token'] >= 0
    for d in result['per_dimension']:
        assert 'dimension' in d
        assert 'logit_contribution' in d


def test_critical_dimensions_top_k(model_and_tokens):
    model, tokens = model_and_tokens
    r3 = critical_dimensions(model, tokens, top_k=3)
    r5 = critical_dimensions(model, tokens, top_k=5)
    assert len(r3['per_dimension']) == 3
    assert len(r5['per_dimension']) == 5


def test_redundancy_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = redundancy_detection(model, tokens, layers=[1])
    assert len(result['per_layer']) == 1
