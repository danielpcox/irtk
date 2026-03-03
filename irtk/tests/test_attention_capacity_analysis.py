"""Tests for attention_capacity_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_capacity_analysis import (
    attention_pattern_rank,
    attention_information_throughput,
    attention_bottleneck_detection,
    head_utilization,
    capacity_allocation,
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


def test_attention_pattern_rank(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_rank(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['effective_rank'] > 0
        assert h['n_significant'] >= 1


def test_attention_information_throughput(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_information_throughput(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['throughput'] >= 0
        assert h['mean_entropy'] >= 0


def test_attention_bottleneck_detection(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_bottleneck_detection(model, tokens)
    assert len(result['per_layer']) == 2
    assert 0 <= result['most_bottlenecked_layer'] < 2
    assert 0 <= result['max_bottleneck_score'] <= 1.0


def test_head_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_utilization(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['selectivity'] >= 0
        assert h['output_norm'] >= 0


def test_capacity_allocation(model_and_tokens):
    model, tokens = model_and_tokens
    result = capacity_allocation(model, tokens)
    assert len(result['head_norms']) == 2
    assert len(result['layer_share']) == 2
    assert result['total_capacity'] > 0


def test_pattern_rank_ratio(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_rank(model, tokens, layer=0)
    for h in result['per_head']:
        assert 0 < h['rank_ratio'] <= 1.0


def test_bottleneck_per_head_ranks(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_bottleneck_detection(model, tokens)
    for layer in result['per_layer']:
        assert len(layer['per_head_ranks']) == 4


def test_capacity_gini(model_and_tokens):
    model, tokens = model_and_tokens
    result = capacity_allocation(model, tokens)
    assert -1.0 <= result['gini_coefficient'] <= 1.0


def test_utilization_layer1(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_utilization(model, tokens, layer=1)
    assert len(result['per_head']) == 4
    assert result['mean_utilization'] >= 0
