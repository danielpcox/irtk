"""Tests for causal_graph_inference module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.causal_graph_inference import (
    component_causal_edges,
    information_bottleneck_detection,
    causal_path_strength,
    critical_component_ordering,
    graph_robustness,
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


def test_component_causal_edges(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_causal_edges(model, tokens)
    assert result['n_edges'] == 4  # 2 layers * (attn + mlp)
    assert result['strongest_edge'] is not None
    for e in result['edges']:
        assert e['strength'] >= 0


def test_information_bottleneck_detection(model_and_tokens):
    model, tokens = model_and_tokens
    result = information_bottleneck_detection(model, tokens)
    assert len(result['per_layer']) == 2
    assert 0 <= result['bottleneck_layer'] < 2
    for layer in result['per_layer']:
        assert layer['effective_dim'] > 0
        assert layer['residual_norm'] >= 0


def test_causal_path_strength(model_and_tokens):
    model, tokens = model_and_tokens
    result = causal_path_strength(model, tokens, source_layer=0, target_layer=1)
    assert result['source_layer'] == 0
    assert result['target_layer'] == 1
    assert result['path_disruption'] >= 0
    assert len(result['per_intermediate']) == 2


def test_critical_component_ordering(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_component_ordering(model, tokens)
    assert len(result['components']) == 10  # 2 layers * (4 heads + 1 MLP)
    assert result['most_critical'] is not None
    for c in result['components']:
        assert c['criticality'] >= 0
        assert 0 <= c['cumulative_fraction'] <= 1.01


def test_graph_robustness(model_and_tokens):
    model, tokens = model_and_tokens
    result = graph_robustness(model, tokens, n_ablations=2)
    assert len(result['ablated_hooks']) == 2
    assert result['joint_effect'] >= 0
    assert 0 <= result['prediction_survival_rate'] <= 1.0


def test_edges_sorted_by_strength(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_causal_edges(model, tokens)
    for i in range(len(result['edges']) - 1):
        assert result['edges'][i]['strength'] >= result['edges'][i+1]['strength'] - 0.01


def test_criticality_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_component_ordering(model, tokens)
    for i in range(len(result['components']) - 1):
        assert result['components'][i]['criticality'] >= result['components'][i+1]['criticality'] - 0.01


def test_cumulative_reaches_one(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_component_ordering(model, tokens)
    assert result['components'][-1]['cumulative_fraction'] > 0.99


def test_bottleneck_dim_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = information_bottleneck_detection(model, tokens)
    assert result['min_effective_dim'] > 0
