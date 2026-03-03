"""Tests for model_compression_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_compression_analysis import (
    head_pruning_candidates,
    layer_pruning_candidates,
    effective_parameter_count,
    weight_low_rank_compressibility,
    redundancy_score,
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


def test_head_pruning_candidates(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pruning_candidates(model, tokens, top_k=3)
    assert len(result['pruning_candidates']) == 3
    assert len(result['all_scores']) == 8
    # Candidates should be sorted by norm (ascending)
    for i in range(len(result['pruning_candidates']) - 1):
        assert result['pruning_candidates'][i]['output_norm'] <= result['pruning_candidates'][i+1]['output_norm'] + 0.01


def test_layer_pruning_candidates(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_pruning_candidates(model, tokens)
    assert len(result['per_layer']) == 2
    assert 0 <= result['most_skippable'] < 2
    assert 0 <= result['least_skippable'] < 2


def test_effective_parameter_count(model_and_tokens):
    model, tokens = model_and_tokens
    result = effective_parameter_count(model, threshold=0.01)
    assert result['total_params'] > 0
    assert result['effective_params'] > 0
    assert 0 < result['compression_ratio'] <= 1.0


def test_weight_low_rank_compressibility(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_low_rank_compressibility(model, layer=0)
    assert 'W_Q' in result['matrices']
    assert 'W_in' in result['matrices']
    for name, m in result['matrices'].items():
        assert m['rank_for_90pct'] > 0
        assert m['rank_for_95pct'] >= m['rank_for_90pct']


def test_redundancy_score(model_and_tokens):
    model, tokens = model_and_tokens
    result = redundancy_score(model, tokens)
    assert -1.0 <= result['redundancy_score'] <= 1.0
    assert result['n_layer_pairs'] == 1
    assert result['n_head_pairs'] == 12  # C(4,2) * 2 layers


def test_effective_params_per_component(model_and_tokens):
    model, tokens = model_and_tokens
    result = effective_parameter_count(model, threshold=0.001)
    assert 'embed' in result['per_component']
    assert 'unembed' in result['per_component']
    assert 'layer_0' in result['per_component']


def test_low_rank_with_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = weight_low_rank_compressibility(model, layer=0, target_rank=2)
    for name, m in result['matrices'].items():
        assert 'energy_at_target' in m
        assert 0 < m['energy_at_target'] <= 1.0


def test_head_pruning_all_scores(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pruning_candidates(model, tokens, top_k=8)
    assert len(result['pruning_candidates']) == 8


def test_layer_contribution_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_pruning_candidates(model, tokens)
    for layer in result['per_layer']:
        assert layer['contribution_norm'] >= 0
        assert layer['relative_contribution'] >= 0
