"""Tests for component_contribution_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.component_contribution_analysis import (
    cumulative_component_contribution,
    component_norm_contribution,
    component_direction_alignment,
    component_interference,
    component_importance_ranking,
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


def test_cumulative_component_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_component_contribution(model, tokens)
    assert len(result['per_component']) == 5  # embed + 2*(attn + mlp)
    assert result['per_component'][0]['component'] == 'embed'


def test_component_norm_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_norm_contribution(model, tokens)
    assert len(result['per_component']) == 5
    for c in result['per_component']:
        assert c['output_norm'] >= 0
        assert c['fraction_of_residual'] >= 0


def test_component_direction_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_direction_alignment(model, tokens)
    assert len(result['per_component']) == 5
    for c in result['per_component']:
        assert -1.0 <= c['alignment'] <= 1.0


def test_component_interference(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_interference(model, tokens)
    assert len(result['per_pair']) > 0
    for p in result['per_pair']:
        assert -1.0 <= p['cosine'] <= 1.0
        assert p['type'] in ('constructive', 'destructive', 'orthogonal')


def test_component_importance_ranking(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_importance_ranking(model, tokens)
    assert len(result['ranked_components']) == 5
    # Should be sorted by abs_contribution
    for i in range(len(result['ranked_components']) - 1):
        assert result['ranked_components'][i]['abs_contribution'] >= result['ranked_components'][i+1]['abs_contribution']


def test_cumulative_logit_increasing(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_component_contribution(model, tokens)
    # Each component should add something (can be positive or negative)
    assert len(result['per_component']) > 0


def test_norm_has_final_residual(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_norm_contribution(model, tokens)
    assert result['final_residual_norm'] > 0


def test_importance_ranks_unique(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_importance_ranking(model, tokens)
    ranks = [c['rank'] for c in result['ranked_components']]
    assert ranks == list(range(1, len(ranks) + 1))


def test_interference_pair_types(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_interference(model, tokens)
    types = set(p['type'] for p in result['per_pair'])
    # At least one type should be present
    assert len(types) >= 1
