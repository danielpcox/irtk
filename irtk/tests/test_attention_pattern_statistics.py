"""Tests for attention_pattern_statistics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_statistics import (
    attention_entropy_profile, attention_concentration_profile,
    attention_positional_bias, attention_pattern_stability,
    attention_head_diversity,
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


def test_entropy_profile_structure(model, tokens):
    result = attention_entropy_profile(model, tokens)
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    assert 'n_focused' in result
    for h in result['per_head']:
        assert isinstance(h['is_focused'], bool)


def test_entropy_profile_values(model, tokens):
    result = attention_entropy_profile(model, tokens)
    for h in result['per_head']:
        assert h['mean_entropy'] >= 0
        assert 0 <= h['normalized_entropy'] <= 1.5  # can slightly exceed 1


def test_concentration_profile_structure(model, tokens):
    result = attention_concentration_profile(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert isinstance(h['is_sharp'], bool)
        assert 0 <= h['mean_top1_mass'] <= 1


def test_positional_bias_structure(model, tokens):
    result = attention_positional_bias(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['dominant_bias'] in ('bos', 'self', 'prev')
        assert 0 <= h['mean_bos_attention'] <= 1


def test_positional_bias_values(model, tokens):
    result = attention_positional_bias(model, tokens)
    for h in result['per_head']:
        assert 0 <= h['mean_self_attention'] <= 1
        assert 0 <= h['mean_prev_attention'] <= 1


def test_pattern_stability_structure(model, tokens):
    result = attention_pattern_stability(model, tokens)
    assert 'n_stable' in result
    for h in result['per_head']:
        assert isinstance(h['is_stable'], bool)
        assert h['mean_consecutive_diff'] >= 0


def test_head_diversity_structure(model, tokens):
    result = attention_head_diversity(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['pairs']) == 6  # C(4,2)
    assert isinstance(result['is_diverse'], bool)


def test_head_diversity_similarity_range(model, tokens):
    result = attention_head_diversity(model, tokens, layer=0)
    for p in result['pairs']:
        assert -1.0 <= p['similarity'] <= 1.0


def test_entropy_all_layers(model, tokens):
    result = attention_entropy_profile(model, tokens)
    layers_seen = set()
    for h in result['per_head']:
        layers_seen.add(h['layer'])
    assert layers_seen == {0, 1}
