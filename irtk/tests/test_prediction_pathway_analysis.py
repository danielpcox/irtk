"""Tests for prediction_pathway_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_pathway_analysis import (
    prediction_buildup, prediction_component_attribution,
    prediction_confidence_evolution, prediction_competition,
    prediction_commit_point,
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


def test_prediction_buildup_structure(model, tokens):
    result = prediction_buildup(model, tokens, position=2)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['target_rank'] >= 0


def test_prediction_buildup_logits(model, tokens):
    result = prediction_buildup(model, tokens)
    for p in result['per_layer']:
        assert p['top_logit'] >= p['target_logit']


def test_prediction_component_attribution_structure(model, tokens):
    result = prediction_component_attribution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert -1 <= p['attn_alignment'] <= 1


def test_prediction_confidence_evolution_structure(model, tokens):
    result = prediction_confidence_evolution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['max_prob'] <= 1
        assert p['entropy'] >= 0


def test_prediction_competition_structure(model, tokens):
    result = prediction_competition(model, tokens, top_k=3)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert len(p['candidates']) == 3
        probs = [c['probability'] for c in p['candidates']]
        assert probs == sorted(probs, reverse=True)


def test_prediction_competition_margin(model, tokens):
    result = prediction_competition(model, tokens, top_k=3)
    for p in result['per_layer']:
        assert p['margin'] >= 0


def test_prediction_commit_point_structure(model, tokens):
    result = prediction_commit_point(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert isinstance(p['matches_final'], bool)


def test_prediction_commit_point_final(model, tokens):
    result = prediction_commit_point(model, tokens)
    # Last layer must match final
    assert result['per_layer'][-1]['matches_final']


def test_prediction_component_total(model, tokens):
    result = prediction_component_attribution(model, tokens)
    for p in result['per_layer']:
        expected = p['attn_logit_contribution'] + p['mlp_logit_contribution']
        assert abs(p['total_contribution'] - expected) < 1e-4
