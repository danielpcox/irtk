"""Tests for embedding_projection_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_projection_analysis import (
    embedding_to_attention_projection,
    embedding_to_mlp_projection,
    embedding_unembed_circuit,
    embedding_subspace_utilization,
    token_embedding_similarity_structure,
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


def test_embedding_to_attention_projection(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_to_attention_projection(model, tokens, layer=0)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert h['q_projection_norm'] >= 0
        assert h['k_projection_norm'] >= 0
        assert h['v_projection_norm'] >= 0


def test_embedding_to_mlp_projection(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_to_mlp_projection(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 0 <= p['activation_fraction'] <= 1.0


def test_embedding_unembed_circuit(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_unembed_circuit(model, tokens)
    assert len(result['per_position']) == 5
    assert 0 <= result['self_prediction_rate'] <= 1.0


def test_embedding_subspace_utilization(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_subspace_utilization(model, tokens)
    assert result['effective_rank'] >= 1.0
    assert 0 <= result['utilization'] <= 1.01
    assert result['dims_for_90_pct'] >= 1


def test_token_embedding_similarity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_embedding_similarity_structure(model, tokens)
    assert -1.0 <= result['mean_content_similarity'] <= 1.01
    assert result['n_pairs'] == 10  # C(5,2)


def test_attention_projection_norms(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_to_attention_projection(model, tokens, layer=0)
    assert result['mean_q_norm'] >= 0
    assert result['mean_v_norm'] >= 0


def test_mlp_activation_stats(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_to_mlp_projection(model, tokens, layer=0)
    assert 0 <= result['mean_activation_fraction'] <= 1.0


def test_unembed_circuit_per_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_unembed_circuit(model, tokens)
    for p in result['per_position']:
        assert p['entropy'] >= 0
        assert 0 <= p['confidence'] <= 1.0


def test_similarity_pairs(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_embedding_similarity_structure(model, tokens)
    assert result['most_similar_pair'] is not None
    assert result['least_similar_pair'] is not None
