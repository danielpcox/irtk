"""Tests for layer_contribution_ranking module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_contribution_ranking import (
    layer_logit_contribution, layer_norm_contribution,
    layer_direction_importance, layer_entropy_contribution,
    layer_cumulative_effect,
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


def test_logit_contribution_structure(model, tokens):
    result = layer_logit_contribution(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'target_token' in result
    for p in result['per_layer']:
        assert 'attn_logit_contrib' in p
        assert 'mlp_logit_contrib' in p


def test_logit_contribution_sorted(model, tokens):
    result = layer_logit_contribution(model, tokens)
    contribs = [abs(p['total_logit_contrib']) for p in result['per_layer']]
    assert contribs == sorted(contribs, reverse=True)


def test_norm_contribution_structure(model, tokens):
    result = layer_norm_contribution(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_norm'] >= 0
        assert p['mlp_norm'] >= 0


def test_norm_contribution_sorted(model, tokens):
    result = layer_norm_contribution(model, tokens)
    norms = [p['total_norm'] for p in result['per_layer']]
    assert norms == sorted(norms, reverse=True)


def test_direction_importance_structure(model, tokens):
    result = layer_direction_importance(model, tokens)
    assert 'n_constructive' in result
    for p in result['per_layer']:
        assert isinstance(p['is_constructive'], bool)
        assert -1 <= p['cosine_with_final'] <= 1


def test_entropy_contribution_structure(model, tokens):
    result = layer_entropy_contribution(model, tokens)
    assert 'n_sharpening' in result
    for p in result['per_layer']:
        assert isinstance(p['sharpens'], bool)
        assert p['pre_entropy'] >= 0
        assert p['post_entropy'] >= 0


def test_cumulative_effect_structure(model, tokens):
    result = layer_cumulative_effect(model, tokens)
    assert 'target_token' in result
    assert len(result['cumulative']) == 2
    for c in result['cumulative']:
        assert isinstance(c['is_top1'], bool)
        assert c['target_rank'] >= 0


def test_cumulative_effect_layers_ordered(model, tokens):
    result = layer_cumulative_effect(model, tokens)
    layers = [c['layer'] for c in result['cumulative']]
    assert layers == sorted(layers)


def test_direction_importance_sorted(model, tokens):
    result = layer_direction_importance(model, tokens)
    projections = [p['projection'] for p in result['per_layer']]
    assert projections == sorted(projections, reverse=True)
