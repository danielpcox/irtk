"""Tests for attention_score_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_score_decomposition import (
    score_magnitude_profile, score_position_decomposition,
    score_temperature_analysis, score_cross_head_comparison,
    score_softmax_saturation,
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


def test_score_magnitude_profile_structure(model, tokens):
    result = score_magnitude_profile(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['score_range'] >= 0


def test_score_position_decomposition_structure(model, tokens):
    result = score_position_decomposition(model, tokens, layer=0, head=0)
    assert result['query_norm'] > 0
    assert len(result['per_key']) > 0


def test_score_position_decomposition_sorted(model, tokens):
    result = score_position_decomposition(model, tokens, layer=0, head=0, position=-1)
    scores = [k['score'] for k in result['per_key']]
    assert scores == sorted(scores, reverse=True)


def test_score_temperature_analysis_structure(model, tokens):
    result = score_temperature_analysis(model, tokens, layer=0, head=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['effective_temperature'] >= 0
        assert 0 <= p['max_probability'] <= 1


def test_score_cross_head_comparison_structure(model, tokens):
    result = score_cross_head_comparison(model, tokens, layer=0)
    assert len(result['pairs']) == 6  # C(4,2)
    for p in result['pairs']:
        assert -1.0 <= p['score_similarity'] <= 1.0


def test_score_softmax_saturation_structure(model, tokens):
    result = score_softmax_saturation(model, tokens)
    assert len(result['per_head']) == 8
    assert 'n_saturated' in result
    for h in result['per_head']:
        assert isinstance(h['is_saturated'], bool)
        assert 0 <= h['mean_max_prob'] <= 1


def test_score_magnitude_values(model, tokens):
    result = score_magnitude_profile(model, tokens)
    for h in result['per_head']:
        assert h['max_score'] >= h['min_score']


def test_score_temperature_entropy(model, tokens):
    result = score_temperature_analysis(model, tokens, layer=0, head=0)
    for p in result['per_position']:
        assert p['entropy'] >= 0


def test_score_decomposition_cosine_range(model, tokens):
    result = score_position_decomposition(model, tokens, layer=0, head=0)
    for k in result['per_key']:
        assert -1.0 <= k['cosine'] <= 1.0
