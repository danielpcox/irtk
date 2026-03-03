"""Tests for cross-layer alignment."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.cross_layer_alignment import (
    adjacent_layer_alignment, full_layer_alignment_matrix,
    component_alignment_across_layers, early_late_alignment,
    cross_layer_alignment_summary,
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
    tokens = jnp.array([1, 5, 10, 15, 20])
    return model, tokens


def test_adjacent_layer_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = adjacent_layer_alignment(model, tokens, position=-1)
    assert "position" in result
    assert "per_pair" in result
    assert "mean_alignment" in result
    assert len(result["per_pair"]) == 1  # 2 layers -> 1 pair


def test_adjacent_layer_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = adjacent_layer_alignment(model, tokens, position=-1)
    for p in result["per_pair"]:
        assert -1.0 <= p["cosine"] <= 1.0
        assert "is_aligned" in p


def test_full_layer_alignment_matrix_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = full_layer_alignment_matrix(model, tokens, position=-1)
    assert "alignment_matrix" in result
    assert "n_layers" in result
    assert result["n_layers"] == 2
    assert len(result["alignment_matrix"]) == 2
    assert len(result["alignment_matrix"][0]) == 2


def test_full_layer_alignment_matrix_diagonal(model_and_tokens):
    model, tokens = model_and_tokens
    result = full_layer_alignment_matrix(model, tokens, position=-1)
    for i in range(2):
        assert abs(result["alignment_matrix"][i][i] - 1.0) < 1e-4


def test_component_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_alignment_across_layers(model, tokens, position=-1)
    assert "attn_alignment" in result
    assert "mlp_alignment" in result
    assert len(result["attn_alignment"]) == 1  # 2 layers -> 1 pair
    assert len(result["mlp_alignment"]) == 1


def test_component_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_alignment_across_layers(model, tokens, position=-1)
    for p in result["attn_alignment"]:
        assert -1.0 <= p["cosine"] <= 1.0
    for p in result["mlp_alignment"]:
        assert -1.0 <= p["cosine"] <= 1.0


def test_early_late_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = early_late_alignment(model, tokens, position=-1)
    assert "early_late_cosine" in result
    assert "early_norm" in result
    assert "late_norm" in result
    assert "norm_growth" in result
    assert "is_preserved" in result


def test_early_late_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = early_late_alignment(model, tokens, position=-1)
    assert -1.0 <= result["early_late_cosine"] <= 1.0
    assert result["early_norm"] > 0
    assert result["late_norm"] > 0
    assert result["norm_growth"] > 0


def test_cross_layer_alignment_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = cross_layer_alignment_summary(model, tokens, position=-1)
    assert "mean_adjacent_alignment" in result
    assert "early_late_cosine" in result
    assert "norm_growth" in result
    assert "is_preserved" in result
