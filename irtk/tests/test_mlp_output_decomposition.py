"""Tests for mlp_output_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_output_decomposition import (
    per_neuron_output_direction,
    position_neuron_contributions,
    neuron_selectivity_profile,
    mlp_output_direction_clustering,
    mlp_residual_alignment,
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


def test_per_neuron_output_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_neuron_output_direction(model, tokens, layer=0, top_k=5)
    assert result['n_analyzed'] == 5
    for n in result['per_neuron']:
        assert n['mean_activation'] >= 0


def test_position_neuron_contributions(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_neuron_contributions(model, tokens, layer=0, position=-1, top_k=3)
    assert result['position'] == 4
    assert len(result['per_neuron']) == 3
    assert result['total_contribution'] >= 0


def test_neuron_selectivity_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_selectivity_profile(model, tokens, layer=0, top_k=5)
    assert len(result['per_neuron']) == 5
    for n in result['per_neuron']:
        assert 0 <= n['selectivity'] <= 1.0


def test_mlp_output_direction_clustering(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_direction_clustering(model, tokens, layer=0, n_clusters=3)
    assert result['n_active'] > 0
    assert len(result['clusters']) == 3


def test_mlp_residual_alignment(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_alignment(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    assert -1.0 <= result['mean_alignment'] <= 1.01
    assert 0 <= result['reinforcement_fraction'] <= 1.0


def test_contribution_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = position_neuron_contributions(model, tokens, layer=0, top_k=3)
    total_frac = sum(n['fraction'] for n in result['per_neuron'])
    assert total_frac <= 1.01


def test_selectivity_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_selectivity_profile(model, tokens, layer=0, top_k=5)
    assert result['n_selective'] >= 0
    assert result['n_selective'] <= 5


def test_clustering_members(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_output_direction_clustering(model, tokens, layer=0, n_clusters=3)
    total_members = sum(c['n_members'] for c in result['clusters'])
    assert total_members == result['n_active']


def test_alignment_per_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_alignment(model, tokens, layer=0)
    for p in result['per_position']:
        assert -1.0 <= p['cosine_alignment'] <= 1.01
        assert isinstance(p['reinforces'], bool)
