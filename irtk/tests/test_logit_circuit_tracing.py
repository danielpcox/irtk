"""Tests for logit_circuit_tracing module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_circuit_tracing import (
    trace_logit_to_components,
    per_head_logit_contribution,
    logit_attribution_path,
    competing_logit_analysis,
    critical_circuit_components,
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


def test_trace_logit_to_components(model_and_tokens):
    model, tokens = model_and_tokens
    result = trace_logit_to_components(model, tokens, target_token=5, position=-1)
    assert result['target_token'] == 5
    assert len(result['components']) > 0
    assert isinstance(result['actual_logit'], float)


def test_per_head_logit_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_head_logit_contribution(model, tokens, target_token=5, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert isinstance(h['promotes'], bool)


def test_logit_attribution_path(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_attribution_path(model, tokens, target_token=5)
    assert len(result['per_layer']) == 2
    assert isinstance(result['final_logit'], float)


def test_competing_logit_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = competing_logit_analysis(model, tokens, position=-1, top_k=3)
    assert len(result['per_token']) == 3
    assert isinstance(result['winner'], int)
    assert result['margin'] >= 0


def test_critical_circuit_components(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_circuit_components(model, tokens, target_token=5)
    assert result['n_critical'] >= 0
    for c in result['components']:
        assert isinstance(c['is_critical'], bool)


def test_trace_component_types(model_and_tokens):
    model, tokens = model_and_tokens
    result = trace_logit_to_components(model, tokens, target_token=5)
    types = set(c['type'] for c in result['components'])
    assert 'attention' in types
    assert 'mlp' in types


def test_path_cumulative(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_attribution_path(model, tokens, target_token=5)
    # Each layer should have cumulative logit
    for p in result['per_layer']:
        assert isinstance(p['cumulative_logit'], float)


def test_competing_margin(model_and_tokens):
    model, tokens = model_and_tokens
    result = competing_logit_analysis(model, tokens, top_k=5)
    assert result['per_token'][0]['final_logit'] >= result['per_token'][1]['final_logit']


def test_critical_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    result = critical_circuit_components(model, tokens, target_token=5, threshold=0.5)
    assert result['n_critical'] <= len(result['components'])
