"""Tests for attention_output_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_output_decomposition import (
    per_source_output_decomposition,
    output_vocabulary_alignment,
    head_logit_effect_profile,
    attention_weighted_value_analysis,
    head_output_direction_stability,
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


def test_per_source_output_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_source_output_decomposition(model, tokens, layer=0, head=0)
    assert len(result['per_source']) == 5
    assert result['top_source'] is not None
    for s in result['per_source']:
        assert s['output_norm'] >= 0


def test_output_vocabulary_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = output_vocabulary_alignment(model, tokens, layer=0, head=0, top_k=3)
    assert len(result['promoted_tokens']) == 3
    assert len(result['suppressed_tokens']) == 3
    assert result['output_norm'] >= 0


def test_head_logit_effect_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_logit_effect_profile(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5


def test_attention_weighted_value_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_weighted_value_analysis(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['attention_entropy'] >= 0


def test_head_output_direction_stability(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_direction_stability(model, tokens, layer=0, head=0)
    assert -1.0 <= result['mean_pairwise_similarity'] <= 1.01
    assert isinstance(result['is_stable'], bool)


def test_source_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_source_output_decomposition(model, tokens, layer=0, head=0)
    for i in range(len(result['per_source']) - 1):
        assert result['per_source'][i]['output_norm'] >= result['per_source'][i+1]['output_norm'] - 0.01


def test_vocab_alignment_promoted_gt_suppressed(model_and_tokens):
    model, tokens = model_and_tokens
    result = output_vocabulary_alignment(model, tokens, layer=0, head=0, top_k=1)
    assert result['promoted_tokens'][0]['logit'] >= result['suppressed_tokens'][0]['logit']


def test_logit_profile_all_positions(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_logit_effect_profile(model, tokens, layer=0, head=0)
    for p in result['per_position']:
        assert p['output_norm'] >= 0
        assert 0 <= p['target_token'] < 50


def test_stability_per_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_direction_stability(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert -1.0 <= p['alignment_to_mean'] <= 1.01
