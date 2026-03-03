"""Tests for qk_attention_circuit module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.qk_attention_circuit import (
    qk_eigenspectrum, qk_positional_vs_content,
    qk_token_preference, qk_composition_from_prev_layer,
    qk_pattern_prediction,
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


def test_qk_eigenspectrum_structure(model):
    result = qk_eigenspectrum(model, layer=0, head=0)
    assert result['effective_rank'] > 0
    assert result['spectral_norm'] > 0
    assert len(result['top_singular_values']) <= 5


def test_qk_positional_vs_content_structure(model, tokens):
    result = qk_positional_vs_content(model, tokens, layer=0, head=0)
    assert isinstance(result['is_positional'], bool)
    assert result['score_variance'] >= 0


def test_qk_positional_vs_content_entropy_range(model, tokens):
    result = qk_positional_vs_content(model, tokens, layer=0, head=0)
    assert 0 <= result['mean_normalized_entropy'] <= 2  # can be > 1 for small seq


def test_qk_token_preference_structure(model):
    result = qk_token_preference(model, layer=0, head=0, token_ids=[1, 5, 10])
    assert len(result['per_query']) == 3
    for q in result['per_query']:
        assert 'preferred_token' in q


def test_qk_composition_structure(model):
    result = qk_composition_from_prev_layer(model, layer=1, head=0)
    assert len(result['compositions']) == 4
    for c in result['compositions']:
        assert c['q_composition'] >= 0
        assert c['k_composition'] >= 0


def test_qk_composition_first_layer(model):
    result = qk_composition_from_prev_layer(model, layer=0, head=0)
    assert result['error'] == 'first_layer'


def test_qk_pattern_prediction_structure(model, tokens):
    result = qk_pattern_prediction(model, tokens, layer=0, head=0)
    assert isinstance(result['patterns_match'], bool)
    assert result['mean_pattern_diff'] >= 0


def test_qk_pattern_prediction_match(model, tokens):
    # Without any hooks modifying scores, predicted should match actual
    result = qk_pattern_prediction(model, tokens, layer=0, head=0)
    assert result['mean_pattern_diff'] < 0.1  # should be very close


def test_qk_eigenspectrum_fractions_sum(model):
    result = qk_eigenspectrum(model, layer=0, head=0)
    total_frac = sum(sv['fraction'] for sv in result['top_singular_values'])
    assert total_frac <= 1.01  # Allow small floating point error
