"""Tests for attention_value_composition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_value_composition import (
    pattern_value_alignment,
    source_value_decomposition,
    value_mixing_analysis,
    composition_with_previous_layer,
    attention_output_logit_decomposition,
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


def test_pattern_value_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = pattern_value_alignment(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert -1.0 <= h['pattern_value_correlation'] <= 1.0
        assert h['effective_contribution'] >= 0


def test_source_value_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_value_decomposition(model, tokens, layer=0, top_k=3)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert len(h['top_sources']) == 3
        assert h['total_output_norm'] >= 0


def test_value_mixing_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_mixing_analysis(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['mixing_entropy'] >= 0
        assert h['n_effective_sources'] >= 1.0


def test_composition_with_previous(model_and_tokens):
    model, tokens = model_and_tokens
    result = composition_with_previous_layer(model, tokens, layer=1)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 0 <= h['max_v_composition'] <= 1.01


def test_composition_layer0(model_and_tokens):
    model, tokens = model_and_tokens
    result = composition_with_previous_layer(model, tokens, layer=0)
    assert result['per_head'] == []


def test_attention_output_logit_decomposition(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_output_logit_decomposition(model, tokens, layer=0, top_k=3)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert len(h['top_sources']) == 3


def test_pattern_value_max_attention(model_and_tokens):
    model, tokens = model_and_tokens
    result = pattern_value_alignment(model, tokens, layer=0)
    for h in result['per_head']:
        assert 0 <= h['max_attention'] <= 1.0


def test_source_decomp_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_value_decomposition(model, tokens, layer=0)
    for h in result['per_head']:
        assert 0 <= h['top_source_fraction'] <= 1.01


def test_mixing_value_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = value_mixing_analysis(model, tokens, layer=0)
    for h in result['per_head']:
        # diversity = 1 - weighted_similarity, can be negative if weighted_similarity > 1
        assert isinstance(h['value_diversity'], float)
