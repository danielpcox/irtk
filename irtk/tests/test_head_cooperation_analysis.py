"""Tests for head_cooperation_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_cooperation_analysis import (
    within_layer_cooperation,
    cross_layer_head_alignment,
    head_redundancy_analysis,
    head_output_interference,
    head_specialization_diversity,
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


def test_within_layer_cooperation(model_and_tokens):
    model, tokens = model_and_tokens
    result = within_layer_cooperation(model, tokens, layer=0, position=-1)
    assert len(result['pairs']) == 6  # C(4,2)
    assert isinstance(result['is_cooperative'], bool)
    assert result['n_cooperative_pairs'] + result['n_competing_pairs'] == 6


def test_cross_layer_head_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_head_alignment(model, tokens, head=0)
    assert len(result['per_layer']) == 2
    assert isinstance(result['is_consistent'], bool)
    assert -1.0 <= result['mean_alignment'] <= 1.01


def test_head_redundancy_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_redundancy_analysis(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    assert result['n_redundant'] >= 0
    for h in result['per_head']:
        assert isinstance(h['is_redundant'], bool)


def test_head_output_interference(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_interference(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['interference_ratio'] >= 0
    assert isinstance(result['has_significant_interference'], bool)


def test_head_specialization_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_specialization_diversity(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    assert 0 <= result['mean_diversity'] <= 1.01
    assert result['n_unique_heads'] >= 0


def test_cooperation_symmetry(model_and_tokens):
    model, tokens = model_and_tokens
    result = within_layer_cooperation(model, tokens, layer=0)
    for p in result['pairs']:
        assert -1.0 <= p['cosine_similarity'] <= 1.01


def test_alignment_norms(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_head_alignment(model, tokens, head=0)
    for p in result['per_layer']:
        assert p['output_norm'] >= 0


def test_interference_ratio_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_interference(model, tokens, layer=0)
    for p in result['per_position']:
        assert p['sum_of_norms'] >= p['norm_of_sum'] - 0.01


def test_diversity_per_head(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_specialization_diversity(model, tokens, layer=0)
    for h in result['per_head']:
        assert h['pattern_entropy'] >= 0
