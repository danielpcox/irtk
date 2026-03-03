"""Tests for attention_pattern_interpretation module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_interpretation import (
    detect_attention_motifs,
    attention_pattern_summary,
    head_function_profile,
    all_heads_motif_classification,
    attention_pattern_evolution,
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


def test_detect_attention_motifs(model_and_tokens):
    model, tokens = model_and_tokens
    result = detect_attention_motifs(model, tokens, layer=0, head=0)
    assert result['dominant_motif'] in ('previous_token', 'self_attention', 'bos_attention', 'uniform')
    assert 0 <= result['diagonal_score'] <= 1.0
    assert 0 <= result['identity_score'] <= 1.0
    assert 0 <= result['bos_score'] <= 1.0


def test_attention_pattern_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_summary(model, tokens, layer=0, head=0)
    assert result['mean_entropy'] >= 0
    assert 0 <= result['sparsity'] <= 1.0
    assert isinstance(result['is_sparse'], bool)


def test_head_function_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_function_profile(model, tokens, layer=0, head=0)
    assert result['mean_output_norm'] >= 0
    assert result['mean_logit_magnitude'] >= 0
    assert -1.0 <= result['direction_consistency'] <= 1.01
    assert isinstance(result['is_consistent'], bool)


def test_all_heads_motif_classification(model_and_tokens):
    model, tokens = model_and_tokens
    result = all_heads_motif_classification(model, tokens)
    assert result['n_heads_total'] == 8  # 2 layers * 4 heads
    assert len(result['per_head']) == 8
    total_motif = sum(result['motif_counts'].values())
    assert total_motif == 8


def test_attention_pattern_evolution(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_evolution(model, tokens, head=0)
    assert len(result['per_layer']) == 2
    assert isinstance(result['is_stable'], bool)
    for p in result['per_layer']:
        assert p['mean_entropy'] >= 0
        assert 0 <= p['similarity_to_previous'] <= 1.01


def test_motif_scores_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = detect_attention_motifs(model, tokens, layer=0, head=0)
    for score in result['motif_scores'].values():
        assert 0 <= score <= 1.01


def test_summary_distance(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_summary(model, tokens, layer=0, head=0)
    assert result['mean_distance'] >= 0


def test_classification_all_heads(model_and_tokens):
    model, tokens = model_and_tokens
    result = all_heads_motif_classification(model, tokens)
    for h in result['per_head']:
        assert 0 <= h['layer'] < 2
        assert 0 <= h['head'] < 4


def test_evolution_stability(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_evolution(model, tokens, head=0)
    assert 0 <= result['mean_stability'] <= 1.01
