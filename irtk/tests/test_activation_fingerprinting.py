"""Tests for activation_fingerprinting module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_fingerprinting import (
    layer_activation_fingerprint, head_output_fingerprint,
    mlp_activation_fingerprint, attention_pattern_fingerprint,
    input_sensitivity_fingerprint,
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


def test_layer_activation_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_activation_fingerprint(model, tokens)
    assert result['n_layers'] == 2
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'mean_norm' in p
        assert p['mean_norm'] >= 0
        assert 'mean_pairwise_similarity' in p


def test_layer_fingerprint_cosine(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_activation_fingerprint(model, tokens)
    for p in result['per_layer']:
        assert -1.01 <= p['cos_first_last'] <= 1.01


def test_head_output_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_fingerprint(model, tokens, layer=0)
    assert result['layer'] == 0
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert 'mean_norm' in h
        assert h['mean_norm'] >= 0
        assert 'direction_consistency' in h


def test_mlp_activation_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_fingerprint(model, tokens)
    assert result['n_layers'] >= 1
    for p in result['per_layer']:
        assert 'sparsity' in p
        assert 0 <= p['sparsity'] <= 1
        assert 'mean_magnitude' in p


def test_attention_pattern_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_fingerprint(model, tokens)
    assert len(result['per_layer']) == 2
    for layer_info in result['per_layer']:
        assert len(layer_info['per_head']) == 4
        for h in layer_info['per_head']:
            assert 'mean_entropy' in h
            assert h['mean_entropy'] >= 0
            assert 'self_attention' in h
            assert 'prev_token_attention' in h
            assert 'bos_attention' in h


def test_attention_fingerprint_bos(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_fingerprint(model, tokens)
    for layer_info in result['per_layer']:
        for h in layer_info['per_head']:
            assert h['bos_attention'] >= 0


def test_input_sensitivity_fingerprint(model_and_tokens):
    model, tokens = model_and_tokens
    result = input_sensitivity_fingerprint(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 'mean_delta_norm' in p
        assert p['mean_delta_norm'] >= 0
        assert 'relative_change' in p
        assert 'direction_preservation' in p


def test_input_sensitivity_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = input_sensitivity_fingerprint(model, tokens)
    for p in result['per_layer']:
        assert -1.01 <= p['direction_preservation'] <= 1.01


def test_head_fingerprint_max_norm(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_fingerprint(model, tokens, layer=1)
    for h in result['per_head']:
        assert h['max_norm'] >= h['mean_norm']
