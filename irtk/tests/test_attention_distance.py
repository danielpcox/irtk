"""Tests for attention_distance module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_distance import (
    mean_attention_distance,
    local_vs_global_heads,
    distance_weighted_flow,
    attention_decay_curve,
    receptive_field,
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


def test_mean_attention_distance(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_attention_distance(model, tokens)
    assert 'per_head' in result
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    for h in result['per_head']:
        assert h['mean_distance'] >= 0
    assert result['overall_mean_distance'] >= 0


def test_mean_distance_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_attention_distance(model, tokens, layers=[0])
    assert len(result['per_head']) == 4


def test_local_vs_global_heads(model_and_tokens):
    model, tokens = model_and_tokens
    result = local_vs_global_heads(model, tokens, local_window=2)
    assert 'per_head' in result
    for h in result['per_head']:
        assert h['classification'] in ['local', 'global', 'mixed']
        assert 0 <= h['local_attention_mass'] <= 1.0
    total = result['n_local'] + result['n_global'] + result['n_mixed']
    assert total == len(result['per_head'])


def test_distance_weighted_flow(model_and_tokens):
    model, tokens = model_and_tokens
    result = distance_weighted_flow(model, tokens, source_pos=0)
    assert result['source_position'] == 0
    assert 'per_layer' in result
    for layer_result in result['per_layer']:
        assert 'flow_by_distance' in layer_result
        assert layer_result['total_attention_received'] >= 0


def test_attention_decay_curve(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_decay_curve(model, tokens, query_pos=-1)
    assert 'per_head' in result
    assert result['query_position'] == 4  # -1 -> 4 for 5 tokens
    for h in result['per_head']:
        assert 'decay_curve' in h
        assert 'half_life' in h
        assert h['max_attention'] >= 0


def test_receptive_field(model_and_tokens):
    model, tokens = model_and_tokens
    result = receptive_field(model, tokens, target_pos=-1, threshold=0.05)
    assert result['target_position'] == 4
    assert 'per_layer' in result
    for layer_result in result['per_layer']:
        assert layer_result['field_width'] >= 0
    assert result['max_field_width'] >= 0


def test_receptive_field_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    result_low = receptive_field(model, tokens, threshold=0.01)
    result_high = receptive_field(model, tokens, threshold=0.5)
    # Lower threshold should give wider receptive field
    assert result_low['max_field_width'] >= result_high['max_field_width']


def test_flow_different_source(model_and_tokens):
    model, tokens = model_and_tokens
    r1 = distance_weighted_flow(model, tokens, source_pos=0)
    r2 = distance_weighted_flow(model, tokens, source_pos=2)
    assert r1['source_position'] != r2['source_position']


def test_decay_curve_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_decay_curve(model, tokens, layers=[1])
    assert all(h['layer'] == 1 for h in result['per_head'])
