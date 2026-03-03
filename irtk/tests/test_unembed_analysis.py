"""Tests for unembed_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.unembed_analysis import (
    unembed_spectrum, unembed_token_norms,
    unembed_direction_clustering, unembed_component_projection,
    unembed_bias_analysis,
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


def test_unembed_spectrum_structure(model):
    result = unembed_spectrum(model)
    assert result['effective_rank'] > 0
    assert result['d_model'] == 16
    assert result['d_vocab'] == 50


def test_unembed_spectrum_dim90(model):
    result = unembed_spectrum(model)
    assert result['dim_for_90_pct'] > 0
    assert result['dim_for_90_pct'] <= result['d_model']


def test_unembed_token_norms_structure(model):
    result = unembed_token_norms(model, token_ids=[1, 5, 10])
    assert len(result['per_token']) == 3
    assert result['global_mean_norm'] > 0


def test_unembed_token_norms_outlier(model):
    result = unembed_token_norms(model)
    for t in result['per_token']:
        assert isinstance(t['is_outlier'], bool)
        assert t['norm'] > 0


def test_unembed_direction_clustering_structure(model):
    result = unembed_direction_clustering(model, token_ids=[1, 5, 10])
    assert len(result['pairs']) == 3  # C(3,2)
    assert isinstance(result['is_well_separated'], bool)


def test_unembed_direction_clustering_cosine_range(model):
    result = unembed_direction_clustering(model, token_ids=[1, 5, 10])
    for p in result['pairs']:
        assert -1.0 <= p['cosine'] <= 1.0


def test_unembed_component_projection_structure(model, tokens):
    result = unembed_component_projection(model, tokens, layer=0)
    assert len(result['per_component']) == 5  # embed + 2*(attn+mlp)
    for c in result['per_component']:
        assert 'top_token' in c
        assert c['logit_range'] >= 0


def test_unembed_bias_analysis_structure(model):
    result = unembed_bias_analysis(model)
    assert isinstance(result['has_significant_bias'], bool)
    assert len(result['top_biased_tokens']) == 5


def test_unembed_bias_analysis_ordering(model):
    result = unembed_bias_analysis(model)
    top_biases = [t['bias'] for t in result['top_biased_tokens']]
    assert top_biases == sorted(top_biases, reverse=True)
