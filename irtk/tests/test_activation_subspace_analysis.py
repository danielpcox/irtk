"""Tests for activation_subspace_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_subspace_analysis import (
    activation_pca,
    subspace_overlap,
    projection_analysis,
    null_space_analysis,
    component_subspace_analysis,
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


def test_activation_pca(model_and_tokens):
    model, tokens = model_and_tokens
    result = activation_pca(model, tokens, layer=0, n_components=3)
    assert len(result['explained_variance']) == 3
    assert len(result['cumulative_variance']) == 3
    assert result['effective_dimensionality'] > 0
    # Variance fractions should be in [0, 1]
    for v in result['explained_variance']:
        assert 0 <= v <= 1.0


def test_subspace_overlap(model_and_tokens):
    model, tokens = model_and_tokens
    result = subspace_overlap(model, tokens)
    assert len(result['per_pair']) >= 1
    for p in result['per_pair']:
        assert 0 <= p['overlap'] <= 1.01
        assert 0 <= p['max_overlap'] <= 1.01


def test_projection_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jnp.ones(16)  # d_model=16
    result = projection_analysis(model, tokens, direction)
    assert len(result['per_layer']) == 2
    for p in result['per_layer']:
        assert len(p['projections']) == 5
        assert p['std_projection'] >= 0


def test_null_space_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = null_space_analysis(model, tokens, layer=0)
    assert result['utilized_dims'] > 0
    assert result['utilized_dims'] + result['null_dims'] == len(result['singular_values'])
    assert 0 <= result['utilization_fraction'] <= 1.0


def test_component_subspace_analysis(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_subspace_analysis(model, tokens, layer=0)
    assert result['attn_rank'] > 0
    assert result['mlp_rank'] > 0
    assert 0 <= result['shared_subspace'] <= 1.01


def test_pca_cumulative_monotone(model_and_tokens):
    model, tokens = model_and_tokens
    result = activation_pca(model, tokens, layer=0, n_components=5)
    for i in range(1, len(result['cumulative_variance'])):
        assert result['cumulative_variance'][i] >= result['cumulative_variance'][i - 1] - 0.01


def test_projection_single_layer(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = projection_analysis(model, tokens, direction, layer=1)
    assert len(result['per_layer']) == 1
    assert result['per_layer'][0]['layer'] == 1


def test_null_space_singular_values_decreasing(model_and_tokens):
    model, tokens = model_and_tokens
    result = null_space_analysis(model, tokens, layer=0)
    svs = result['singular_values']
    for i in range(len(svs) - 1):
        assert svs[i] >= svs[i + 1] - 0.01


def test_component_subspace_orthogonal_fraction(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_subspace_analysis(model, tokens, layer=0)
    assert result['orthogonal_fraction'] >= 0
