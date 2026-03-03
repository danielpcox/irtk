"""Tests for attention_bottleneck_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_bottleneck_analysis import (
    attention_rank_bottleneck, attention_information_throughput,
    attention_source_concentration, attention_layer_bottleneck,
    attention_position_bottleneck,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([1, 5, 10, 15, 20])


def test_attention_rank_bottleneck_structure(model, tokens):
    result = attention_rank_bottleneck(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['effective_rank'] > 0


def test_attention_rank_bottleneck_sv_fraction(model, tokens):
    result = attention_rank_bottleneck(model, tokens)
    for h in result['per_head']:
        assert 0 <= h['top_sv_fraction'] <= 1
        assert isinstance(h['is_bottleneck'], bool)


def test_attention_information_throughput_structure(model, tokens):
    result = attention_information_throughput(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['mean_throughput'] >= 0
        assert h['max_throughput'] >= h['mean_throughput']


def test_attention_source_concentration_structure(model, tokens):
    result = attention_source_concentration(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['mean_entropy'] >= 0
        assert 0 <= h['mean_top1_mass'] <= 1


def test_attention_source_concentration_normalized(model, tokens):
    result = attention_source_concentration(model, tokens)
    for h in result['per_head']:
        assert 0 <= h['normalized_entropy'] <= 1.01
        assert isinstance(h['is_concentrated'], bool)


def test_attention_layer_bottleneck_structure(model, tokens):
    result = attention_layer_bottleneck(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_pattern_rank'] > 0
        assert isinstance(p['has_bottleneck_head'], bool)


def test_attention_position_bottleneck_structure(model, tokens):
    result = attention_position_bottleneck(model, tokens)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['mean_entropy'] >= 0
        assert isinstance(p['is_bottlenecked'], bool)


def test_attention_position_bottleneck_count(model, tokens):
    result = attention_position_bottleneck(model, tokens)
    assert result['n_bottlenecked'] >= 0
    assert result['n_bottlenecked'] <= 5


def test_attention_throughput_values(model, tokens):
    result = attention_information_throughput(model, tokens)
    for h in result['per_head']:
        assert h['mean_value_norm'] > 0
