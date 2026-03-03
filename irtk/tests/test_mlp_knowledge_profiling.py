"""Tests for mlp_knowledge_profiling module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_knowledge_profiling import (
    neuron_vocabulary_profile, neuron_selectivity_profile,
    neuron_position_specificity, neuron_cooperation_profile,
    layer_knowledge_summary,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([1, 5, 10, 15, 20])


def test_neuron_vocabulary_profile_structure(model):
    result = neuron_vocabulary_profile(model, layer=0, top_k=5)
    assert len(result['per_neuron']) == 5
    for n in result['per_neuron']:
        assert n['output_norm'] > 0
        assert n['logit_range'] >= 0


def test_neuron_vocabulary_logit_order(model):
    result = neuron_vocabulary_profile(model, layer=0, top_k=5)
    for n in result['per_neuron']:
        assert n['top_promoted_logit'] >= n['top_suppressed_logit']


def test_neuron_selectivity_profile_structure(model, tokens):
    result = neuron_selectivity_profile(model, tokens, layer=0, top_k=5)
    assert result['n_ever_active'] >= 0
    assert result['n_dead'] >= 0


def test_neuron_selectivity_rates(model, tokens):
    result = neuron_selectivity_profile(model, tokens, layer=0, top_k=5)
    for n in result['per_neuron']:
        assert 0 <= n['activation_rate'] <= 1
        assert isinstance(n['is_selective'], bool)


def test_neuron_position_specificity_structure(model, tokens):
    result = neuron_position_specificity(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert p['n_active'] >= 0
        assert 0 <= p['sparsity'] <= 1


def test_neuron_cooperation_profile_structure(model, tokens):
    result = neuron_cooperation_profile(model, tokens, layer=0, sample_size=20)
    assert result['n_cooperative_pairs'] >= 0
    assert result['n_sampled'] == 20


def test_neuron_cooperation_jaccard(model, tokens):
    result = neuron_cooperation_profile(model, tokens, layer=0, sample_size=20)
    for p in result['pairs']:
        assert 0 <= p['jaccard'] <= 1


def test_layer_knowledge_summary_structure(model, tokens):
    result = layer_knowledge_summary(model, tokens)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert p['mean_active_neurons'] >= 0
        assert p['logit_impact'] >= 0


def test_layer_knowledge_summary_values(model, tokens):
    result = layer_knowledge_summary(model, tokens)
    for p in result['per_layer']:
        assert p['mean_activation_magnitude'] >= 0
