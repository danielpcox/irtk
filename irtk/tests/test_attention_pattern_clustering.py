"""Tests for attention_pattern_clustering module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_clustering import (
    pattern_archetypes,
    head_pattern_similarity,
    pattern_diversity,
    pattern_stability_across_inputs,
    cross_layer_pattern_evolution,
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


def test_pattern_archetypes(model_and_tokens):
    model, tokens = model_and_tokens
    result = pattern_archetypes(model, tokens, n_clusters=3)
    assert result['n_total_heads'] == 8  # 2 layers × 4 heads
    assert len(result['clusters']) > 0
    total_members = sum(c['n_members'] for c in result['clusters'])
    assert total_members == 8


def test_head_pattern_similarity(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens)
    assert len(result['head_labels']) == 8
    assert len(result['similarity_matrix']) == 8


def test_head_pattern_similarity_single_layer(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens, layer=0)
    assert len(result['head_labels']) == 4
    assert len(result['similarity_matrix']) == 4


def test_pattern_diversity(model_and_tokens):
    model, tokens = model_and_tokens
    result = pattern_diversity(model, tokens)
    assert len(result['per_layer']) == 2
    for layer in result['per_layer']:
        assert -1.0 <= layer['mean_pairwise_similarity'] <= 1.0
        assert -1.0 <= layer['diversity'] <= 2.0


def test_pattern_stability(model_and_tokens):
    model, tokens = model_and_tokens
    tokens2 = jnp.array([5, 15, 25, 35, 45])
    result = pattern_stability_across_inputs(model, tokens, tokens2)
    assert result['n_total'] == 8
    assert 0 <= result['stability_fraction'] <= 1.0


def test_cross_layer_evolution(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_pattern_evolution(model, tokens)
    assert len(result['per_head']) == 4
    for h in result['per_head']:
        assert len(h['layer_transitions']) == 1  # 2 layers -> 1 transition


def test_similarity_diagonal(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_pattern_similarity(model, tokens)
    # Diagonal should be 1.0 (self-similarity)
    for i in range(len(result['similarity_matrix'])):
        assert abs(result['similarity_matrix'][i][i] - 1.0) < 0.01


def test_archetypes_cluster_stats(model_and_tokens):
    model, tokens = model_and_tokens
    result = pattern_archetypes(model, tokens, n_clusters=2)
    for c in result['clusters']:
        assert c['n_members'] > 0
        assert c['mean_entropy'] >= 0


def test_evolution_mean_change(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_pattern_evolution(model, tokens)
    for h in result['per_head']:
        assert h['mean_change'] >= 0
