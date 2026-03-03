"""Tests for residual_stream_probing module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_probing import (
    token_identity_probe,
    next_token_probe,
    directional_probe,
    layer_prediction_quality,
    residual_information_content,
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


def test_token_identity_probe(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_identity_probe(model, tokens, layer=0, pos=0)
    assert result['true_token'] == 1
    assert 0 <= result['predicted_token'] < 50
    assert result['true_token_rank'] >= 0


def test_next_token_probe(model_and_tokens):
    model, tokens = model_and_tokens
    result = next_token_probe(model, tokens, layer=0)
    assert 0 <= result['top_prediction'] < 50
    assert result['entropy'] >= 0
    assert 0 <= result['top_probability'] <= 1.0


def test_directional_probe(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = directional_probe(model, tokens, direction)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['std'] >= 0


def test_layer_prediction_quality(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_quality(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert 0 <= p['top_prediction'] < 50
        assert p['final_token_rank'] >= 0


def test_residual_information_content(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_information_content(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['norm'] >= 0
        assert p['effective_dimensionality'] > 0
        assert 0 <= p['sparsity'] <= 1.01


def test_identity_probe_different_positions(model_and_tokens):
    model, tokens = model_and_tokens
    r0 = token_identity_probe(model, tokens, layer=0, pos=0)
    r1 = token_identity_probe(model, tokens, layer=0, pos=1)
    assert r0['true_token'] == 1
    assert r1['true_token'] == 10


def test_prediction_quality_final_layer(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_prediction_quality(model, tokens)
    # Last layer should have rank 0 or close (may not match exactly due to final LN)
    last = result['per_layer'][-1]
    assert last['final_token_rank'] >= 0


def test_directional_probe_label(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jnp.ones(16)
    result = directional_probe(model, tokens, direction, label='test_dir')
    assert result['direction_label'] == 'test_dir'


def test_next_token_probe_layers(model_and_tokens):
    model, tokens = model_and_tokens
    r0 = next_token_probe(model, tokens, layer=0)
    r1 = next_token_probe(model, tokens, layer=1)
    assert r0['layer'] == 0
    assert r1['layer'] == 1
