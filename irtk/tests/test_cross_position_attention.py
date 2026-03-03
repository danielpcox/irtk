"""Tests for cross_position_attention module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.cross_position_attention import (
    position_information_flow, source_position_importance, attention_flow_matrix,
    position_pair_interaction, attention_bottleneck_positions,
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


def test_position_information_flow(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_information_flow(model, tokens, layer=0)
    assert result['layer'] == 0
    assert 'per_position' in result
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 'position' in p
        assert 'total_received' in p
        assert 'total_sent' in p
        assert isinstance(p['is_hub'], bool)


def test_position_information_flow_layer1(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_information_flow(model, tokens, layer=1)
    assert result['layer'] == 1
    assert 'n_hubs' in result


def test_source_position_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_position_importance(model, tokens, layer=0, target_position=-1)
    assert result['layer'] == 0
    assert result['target_position'] == 4
    assert 'per_source' in result
    assert len(result['per_source']) == 5
    assert 'top_source' in result


def test_source_position_importance_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_position_importance(model, tokens, layer=0)
    attns = [s['total_attention'] for s in result['per_source']]
    assert attns == sorted(attns, reverse=True)


def test_attention_flow_matrix(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_flow_matrix(model, tokens)
    assert 'per_position' in result
    assert len(result['per_position']) == 5
    assert 'mean_flow' in result
    for p in result['per_position']:
        assert 'total_incoming_flow' in p
        assert 'primary_source' in p


def test_position_pair_interaction(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_pair_interaction(model, tokens, pos_a=1, pos_b=3)
    assert result['pos_a'] == 1
    assert result['pos_b'] == 3
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    assert 'total_attention' in result
    for layer_info in result['per_layer']:
        assert 'per_head' in layer_info
        assert len(layer_info['per_head']) == 4


def test_position_pair_interaction_swap(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_pair_interaction(model, tokens, pos_a=3, pos_b=1)
    assert result['pos_a'] == 1
    assert result['pos_b'] == 3


def test_attention_bottleneck_positions(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_bottleneck_positions(model, tokens)
    assert 'per_position' in result
    assert len(result['per_position']) == 5
    assert 'n_bottlenecks' in result
    assert 'top_bottleneck' in result
    for p in result['per_position']:
        assert 'incoming_attention' in p
        assert isinstance(p['is_bottleneck'], bool)


def test_attention_bottleneck_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_bottleneck_positions(model, tokens)
    attns = [p['incoming_attention'] for p in result['per_position']]
    assert attns == sorted(attns, reverse=True)
