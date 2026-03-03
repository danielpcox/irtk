"""Tests for mlp_gate_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_gate_analysis import (
    mlp_activation_distribution,
    mlp_pre_post_relationship,
    neuron_activation_frequency,
    mlp_output_direction_analysis,
    mlp_contribution_vs_attention,
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


def test_mlp_activation_distribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_distribution(model, tokens, layer=0)
    assert 0 <= result['active_fraction'] <= 1.0
    assert result['n_dead_neurons'] >= 0


def test_mlp_pre_post_relationship(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_pre_post_relationship(model, tokens, layer=0)
    assert -1.0 <= result['correlation'] <= 1.01
    assert 0 <= result['pass_rate'] <= 1.0


def test_neuron_activation_frequency(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_frequency(model, tokens, layer=0, top_k=5)
    assert len(result['most_active']) == 5
    assert result['mean_frequency'] >= 0


def test_mlp_output_direction_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_direction_analysis(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    assert result['mean_norm'] >= 0
    assert isinstance(result['is_consistent'], bool)


def test_mlp_contribution_vs_attention(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_vs_attention(model, tokens)
    assert len(result['per_layer']) == 2
    assert result['n_mlp_dominant'] + result['n_attn_dominant'] == 2


def test_activation_ranges(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_distribution(model, tokens, layer=0)
    assert result['max_activation'] >= result['mean_activation']


def test_frequency_counts(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_activation_frequency(model, tokens, layer=0)
    assert result['n_always_active'] >= 0
    assert result['n_never_active'] >= 0


def test_direction_alignment_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_direction_analysis(model, tokens, layer=0)
    for p in result['per_position']:
        assert -1.0 <= p['alignment_to_mean'] <= 1.01


def test_contribution_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_vs_attention(model, tokens)
    for p in result['per_layer']:
        assert 0 <= p['mlp_fraction'] <= 1.0
