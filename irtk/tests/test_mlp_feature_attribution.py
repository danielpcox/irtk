"""Tests for mlp_feature_attribution module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_feature_attribution import (
    neuron_logit_attribution,
    active_neuron_profile,
    mlp_layer_attribution,
    neuron_feature_directions,
    neuron_cooperation,
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


def test_neuron_logit_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_attribution(model, tokens, layer=0, top_k=5)
    assert result['layer'] == 0
    assert 'target_token' in result
    total_neurons = len(result['promoting']) + len(result['suppressing'])
    assert total_neurons <= 5  # top_k
    assert 0 <= result['top_k_fraction'] <= 1.0


def test_neuron_logit_attribution_layer1(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_attribution(model, tokens, layer=1)
    assert result['layer'] == 1


def test_neuron_logit_target_token(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_attribution(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_active_neuron_profile(model_and_tokens):
    model, tokens = model_and_tokens
    result = active_neuron_profile(model, tokens, layer=0)
    assert result['layer'] == 0
    assert 0 <= result['sparsity'] <= 1.0
    assert result['mean_active_per_position'] >= 0
    assert result['max_activation'] >= 0
    assert result['never_active'] >= 0
    assert result['always_active'] >= 0


def test_mlp_layer_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_attribution(model, tokens)
    assert len(result['per_layer']) == 2
    assert 'target_token' in result
    total = sum(p['logit_contribution'] for p in result['per_layer'])
    assert abs(total - result['total_mlp_logit']) < 0.01


def test_mlp_layer_attribution_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_attribution(model, tokens, target_token=10)
    assert result['target_token'] == 10


def test_neuron_feature_directions(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_feature_directions(model, layer=0, top_k=3)
    assert result['layer'] == 0
    assert len(result['per_neuron']) == 3
    for n in result['per_neuron']:
        assert n['output_norm'] > 0
        assert len(n['top_promoted_tokens']) == 3
        assert len(n['top_suppressed_tokens']) == 3


def test_neuron_cooperation(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_cooperation(model, tokens, layer=0, top_k=3)
    assert result['layer'] == 0
    assert result['n_active'] >= 0
    for pair in result['cooperating_pairs']:
        assert pair['cosine_similarity'] > 0.5
    for pair in result['competing_pairs']:
        assert pair['cosine_similarity'] < -0.5


def test_active_profile_threshold(model_and_tokens):
    model, tokens = model_and_tokens
    low = active_neuron_profile(model, tokens, activation_threshold=0.001)
    high = active_neuron_profile(model, tokens, activation_threshold=1.0)
    assert low['sparsity'] <= high['sparsity']
