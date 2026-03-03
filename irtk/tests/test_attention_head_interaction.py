"""Tests for attention_head_interaction module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_interaction import (
    within_layer_interaction,
    cross_layer_alignment,
    attention_pattern_overlap,
    head_output_norms,
    head_pair_importance,
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


def test_within_layer_interaction(model_and_tokens):
    model, tokens = model_and_tokens
    result = within_layer_interaction(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for layer_info in result['per_layer']:
        # C(4,2) = 6 pairs
        assert len(layer_info['interactions']) == 6
        for i in layer_info['interactions']:
            assert -1.0 <= i['cosine_similarity'] <= 1.0
            assert i['type'] in ['cooperative', 'opposing', 'independent']


def test_cross_layer_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_alignment(model, tokens)
    assert 'aligned_pairs' in result
    assert result['n_aligned'] == result['n_reinforcing'] + result['n_canceling']
    for p in result['aligned_pairs']:
        assert abs(p['cosine_similarity']) > 0.5


def test_attention_pattern_overlap(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_overlap(model, tokens)
    assert 'per_layer' in result
    assert len(result['per_layer']) == 2
    for layer_info in result['per_layer']:
        assert layer_info['mean_overlap'] >= 0


def test_attention_pattern_overlap_specific_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_overlap(model, tokens, layers=[0])
    assert len(result['per_layer']) == 1


def test_head_output_norms(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_norms(model, tokens)
    assert len(result['per_head']) == 8  # 2 layers * 4 heads
    for h in result['per_head']:
        assert h['output_norm'] >= 0
        assert 0 <= h['relative_norm'] <= 1.01  # Allow float rounding
    # Sorted by norm descending
    norms = [h['output_norm'] for h in result['per_head']]
    assert norms == sorted(norms, reverse=True)


def test_head_pair_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pair_importance(model, tokens)
    assert 'target_token' in result
    assert len(result['top_pairs']) <= 10
    for p in result['top_pairs']:
        assert isinstance(p['cooperative'], bool)


def test_head_pair_importance_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pair_importance(model, tokens, target_token=3)
    assert result['target_token'] == 3


def test_within_layer_counts(model_and_tokens):
    model, tokens = model_and_tokens
    result = within_layer_interaction(model, tokens)
    for layer_info in result['per_layer']:
        total = layer_info['n_cooperative'] + layer_info['n_opposing'] + layer_info['n_independent']
        assert total == 6


def test_head_norms_dominant(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_norms(model, tokens)
    assert result['dominant_head'] is not None
    assert result['max_norm'] > 0
