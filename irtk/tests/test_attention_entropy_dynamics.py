"""Tests for attention_entropy_dynamics module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_entropy_dynamics import (
    entropy_by_layer, entropy_by_position,
    entropy_head_evolution, entropy_sharpening_profile,
    entropy_diversity_across_heads,
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


def test_entropy_by_layer_structure(model, tokens):
    result = entropy_by_layer(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_entropy'] >= 0


def test_entropy_by_layer_normalized(model, tokens):
    result = entropy_by_layer(model, tokens)
    for p in result['per_layer']:
        assert p['normalized_entropy'] >= 0


def test_entropy_by_position_structure(model, tokens):
    result = entropy_by_position(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['mean_entropy'] >= 0
        assert 0 <= p['normalized_entropy'] <= 1.01


def test_entropy_head_evolution_structure(model, tokens):
    result = entropy_head_evolution(model, tokens, head=0)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_entropy'] >= 0
        assert isinstance(p['is_sharp'], bool)


def test_entropy_head_evolution_max_prob(model, tokens):
    result = entropy_head_evolution(model, tokens, head=0)
    for p in result['per_layer']:
        assert 0 <= p['mean_max_prob'] <= 1


def test_entropy_sharpening_profile_structure(model, tokens):
    result = entropy_sharpening_profile(model, tokens)
    assert isinstance(result['is_sharpening'], bool)
    assert len(result['layer_entropies']) == 2


def test_entropy_sharpening_halves(model, tokens):
    result = entropy_sharpening_profile(model, tokens)
    assert result['first_half_mean'] >= 0
    assert result['second_half_mean'] >= 0


def test_entropy_diversity_structure(model, tokens):
    result = entropy_diversity_across_heads(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['std_entropy'] >= 0
        assert isinstance(p['is_diverse'], bool)


def test_entropy_diversity_range(model, tokens):
    result = entropy_diversity_across_heads(model, tokens)
    for p in result['per_layer']:
        assert p['entropy_range'] >= 0
