"""Tests for attention_value_routing module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_value_routing import (
    value_source_routing, value_output_decomposition,
    value_routing_diversity, value_logit_routing,
    value_routing_summary,
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


def test_value_source_routing_structure(model, tokens):
    result = value_source_routing(model, tokens, layer=0, head=0)
    assert len(result['per_source']) > 0
    for s in result['per_source']:
        assert s['attention_weight'] >= 0


def test_value_source_routing_sorted(model, tokens):
    result = value_source_routing(model, tokens, layer=0, head=0)
    contribs = [s['weighted_contribution'] for s in result['per_source']]
    assert contribs == sorted(contribs, reverse=True)


def test_value_output_decomposition_structure(model, tokens):
    result = value_output_decomposition(model, tokens, layer=0, head=0)
    assert result['total_output_norm'] >= 0
    assert len(result['per_source']) > 0


def test_value_routing_diversity_structure(model, tokens):
    result = value_routing_diversity(model, tokens, layer=0)
    assert isinstance(result['is_diverse'], bool)
    assert len(result['pairs']) == 6  # C(4,2)


def test_value_routing_diversity_cosine(model, tokens):
    result = value_routing_diversity(model, tokens, layer=0)
    for p in result['pairs']:
        assert -1 <= p['output_cosine'] <= 1


def test_value_logit_routing_structure(model, tokens):
    result = value_logit_routing(model, tokens, layer=0, head=0)
    assert result['top_promoted_logit'] >= result['top_suppressed_logit']
    assert result['logit_range'] >= 0


def test_value_routing_summary_structure(model, tokens):
    result = value_routing_summary(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_output_norm'] >= 0
        assert 0 <= p['dominant_head'] < 4


def test_value_routing_summary_max(model, tokens):
    result = value_routing_summary(model, tokens)
    for p in result['per_layer']:
        assert p['max_output_norm'] >= p['mean_output_norm']


def test_value_output_decomp_alignment(model, tokens):
    result = value_output_decomposition(model, tokens, layer=0, head=0)
    for s in result['per_source']:
        assert -1 <= s['alignment_with_total'] <= 1
