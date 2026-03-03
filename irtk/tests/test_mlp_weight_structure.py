"""Tests for mlp_weight_structure module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_weight_structure import (
    mlp_weight_spectrum, mlp_neuron_norms,
    mlp_in_out_alignment, mlp_cross_layer_similarity,
    mlp_unembed_alignment,
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


def test_mlp_weight_spectrum_structure(model):
    result = mlp_weight_spectrum(model, layer=0)
    assert result['W_in_effective_rank'] > 0
    assert result['W_out_effective_rank'] > 0


def test_mlp_weight_spectrum_fractions(model):
    result = mlp_weight_spectrum(model, layer=0)
    assert 0 <= result['W_in_top_sv_fraction'] <= 1
    assert 0 <= result['W_out_top_sv_fraction'] <= 1


def test_mlp_neuron_norms_structure(model):
    result = mlp_neuron_norms(model, layer=0, top_k=5)
    assert len(result['top_neurons']) == 5
    assert result['mean_in_norm'] > 0


def test_mlp_neuron_norms_sorted(model):
    result = mlp_neuron_norms(model, layer=0, top_k=5)
    norms = [n['combined_norm'] for n in result['top_neurons']]
    assert norms == sorted(norms, reverse=True)


def test_mlp_in_out_alignment_structure(model):
    result = mlp_in_out_alignment(model, layer=0)
    assert -1 <= result['mean_alignment'] <= 1
    assert 0 <= result['fraction_aligned'] <= 1


def test_mlp_cross_layer_similarity_structure(model):
    result = mlp_cross_layer_similarity(model)
    assert len(result['pairs']) == 1  # C(2,1)
    for p in result['pairs']:
        assert -1 <= p['W_in_similarity'] <= 1
        assert -1 <= p['W_out_similarity'] <= 1


def test_mlp_unembed_alignment_structure(model):
    result = mlp_unembed_alignment(model, layer=0, top_k=5)
    assert len(result['per_neuron']) == 5
    for n in result['per_neuron']:
        assert n['output_norm'] > 0
        assert 'top_promoted_token' in n


def test_mlp_unembed_alignment_logits(model):
    result = mlp_unembed_alignment(model, layer=0, top_k=5)
    for n in result['per_neuron']:
        assert n['top_promoted_logit'] >= n['top_suppressed_logit']


def test_mlp_neuron_norms_total(model):
    result = mlp_neuron_norms(model, layer=0)
    assert result['d_mlp'] > 0
    assert result['mean_out_norm'] > 0
