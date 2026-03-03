"""Tests for residual_stream_projection module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_projection import (
    project_to_token_direction,
    project_to_difference_direction,
    residual_pca_projection,
    project_to_embedding_subspace,
    multi_direction_projection,
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


def test_project_to_token_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_token_direction(model, tokens, target_token=5)
    assert len(result['per_layer']) == 2
    assert result['target_token'] == 5
    assert isinstance(result['final_projection'], float)


def test_project_to_difference_direction(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_difference_direction(model, tokens, token_a=5, token_b=10)
    assert len(result['per_layer']) == 2
    assert result['final_preference'] in ('a', 'b')


def test_residual_pca_projection(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_pca_projection(model, tokens, layer=0, n_components=3)
    assert len(result['per_component']) == 3
    assert result['total_variance_explained'] > 0


def test_project_to_embedding_subspace(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_embedding_subspace(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 0 <= p['fraction_in_embed'] <= 1.01


def test_multi_direction_projection(model_and_tokens):
    model, tokens = model_and_tokens
    result = multi_direction_projection(model, tokens, target_tokens=[5, 10, 15])
    assert len(result['per_layer']) == 2
    assert result['final_winner'] in [5, 10, 15]


def test_token_direction_per_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_token_direction(model, tokens, target_token=5)
    for p in result['per_layer']:
        assert len(p['per_position']) == 5


def test_pca_variance(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_pca_projection(model, tokens, layer=0)
    assert result['total_variance_explained'] <= 1.01


def test_difference_direction_layers(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_difference_direction(model, tokens, token_a=5, token_b=10)
    for p in result['per_layer']:
        assert isinstance(p['favors_a'], bool)


def test_embedding_subspace_moved(model_and_tokens):
    model, tokens = model_and_tokens
    result = project_to_embedding_subspace(model, tokens, layer=0)
    assert isinstance(result['moved_beyond_embed'], bool)
