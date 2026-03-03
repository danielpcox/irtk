"""Tests for model_architecture_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_architecture_profiling import (
    parameter_count_profile, computation_flow_profile,
    hook_point_inventory, model_capacity_utilization,
    model_summary,
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


def test_parameter_count_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = parameter_count_profile(model)
    assert result['total_parameters'] > 0
    assert result['n_layers'] == 2
    assert 0 <= result['embed_fraction'] <= 1
    assert 0 <= result['layer_fraction'] <= 1
    assert result['embedding_params'] > 0
    assert result['per_layer_attn'] > 0
    assert result['per_layer_mlp'] > 0


def test_parameter_count_sum(model_and_tokens):
    model, tokens = model_and_tokens
    result = parameter_count_profile(model)
    approx_total = (
        result['embedding_params'] + result['positional_params'] +
        result['unembed_params'] + result['total_layer_params']
    )
    assert result['total_parameters'] == approx_total


def test_computation_flow_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_flow_profile(model, tokens)
    assert result['embed_norm'] >= 0
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['attn_output_norm'] >= 0
        assert p['mlp_output_norm'] >= 0
        assert p['residual_norm'] >= 0
        assert 0 <= p['attn_fraction'] <= 1


def test_hook_point_inventory(model_and_tokens):
    model, tokens = model_and_tokens
    result = hook_point_inventory(model)
    assert result['total_hooks'] > 0
    assert result['n_embed'] == 2
    assert result['n_residual'] > 0
    assert result['n_attention'] > 0
    assert result['n_mlp'] > 0
    total = result['n_embed'] + result['n_residual'] + result['n_attention'] + result['n_mlp']
    assert total == result['total_hooks']


def test_hook_inventory_names(model_and_tokens):
    model, tokens = model_and_tokens
    result = hook_point_inventory(model)
    names = [h['name'] for h in result['hook_points']]
    assert 'hook_embed' in names
    assert 'blocks.0.hook_resid_pre' in names
    assert 'blocks.0.attn.hook_pattern' in names


def test_model_capacity_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_capacity_utilization(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_ov_rank'] > 0
        assert p['mlp_effective_rank'] > 0


def test_model_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_summary(model)
    assert result['n_layers'] == 2
    assert result['d_model'] == 16
    assert result['n_heads'] == 4
    assert result['d_head'] == 4
    assert result['d_vocab'] == 50
    assert result['n_ctx'] == 32


def test_model_summary_act_fn(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_summary(model)
    assert 'act_fn' in result
    assert 'normalization_type' in result


def test_computation_flow_monotonic(model_and_tokens):
    model, tokens = model_and_tokens
    result = computation_flow_profile(model, tokens)
    # Residual norms should generally be positive
    for p in result['per_layer']:
        assert p['residual_norm'] > 0
