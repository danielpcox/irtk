"""Tests for mlp_contribution_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_contribution_profiling import (
    mlp_residual_contribution, mlp_logit_effect,
    mlp_position_profile, mlp_layer_comparison,
    mlp_neuron_contribution_ranking,
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


def test_mlp_residual_contribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_contribution(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_constructive' in result
    for p in result['per_layer']:
        assert p['mlp_norm'] >= 0
        assert p['residual_norm'] >= 0
        assert isinstance(p['is_constructive'], bool)


def test_mlp_residual_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_contribution(model, tokens)
    for p in result['per_layer']:
        assert p['fraction_of_residual'] >= 0


def test_mlp_logit_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_logit_effect(model, tokens, position=-1)
    assert result['position'] == 4
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['logit_norm'] >= 0
        assert len(p['top_promoted']) == 5
        assert len(p['top_suppressed']) == 5


def test_mlp_logit_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_logit_effect(model, tokens)
    norms = [p['logit_norm'] for p in result['per_layer']]
    assert norms == sorted(norms, reverse=True)


def test_mlp_position_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_position_profile(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_position']) == 5
    assert 'norm_cv' in result
    assert isinstance(result['is_position_uniform'], bool)
    for p in result['per_position']:
        assert p['norm'] >= 0
        assert isinstance(p['is_aligned'], bool)


def test_mlp_layer_comparison(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_comparison(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'n_unique' in result
    for p in result['per_layer']:
        assert p['mean_norm'] >= 0
        assert isinstance(p['is_unique'], bool)


def test_mlp_neuron_contribution_ranking(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_contribution_ranking(model, tokens, layer=0, top_k=5)
    assert result['layer'] == 0
    assert len(result['per_neuron']) == 5
    assert 'n_active' in result
    assert result['total_neurons'] > 0
    for n in result['per_neuron']:
        assert 'neuron' in n
        assert 'activation' in n
        assert 'contribution' in n
        assert n['contribution'] >= 0


def test_neuron_ranking_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_contribution_ranking(model, tokens, layer=0, top_k=5)
    contribs = [n['contribution'] for n in result['per_neuron']]
    assert contribs == sorted(contribs, reverse=True)


def test_mlp_neuron_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_neuron_contribution_ranking(model, tokens, layer=1, position=2)
    assert result['position'] == 2
