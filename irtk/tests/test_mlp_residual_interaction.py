"""Tests for mlp_residual_interaction module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_residual_interaction import (
    mlp_residual_alignment, mlp_contribution_ratio,
    mlp_residual_decomposition, mlp_vs_attention_balance,
    mlp_residual_summary,
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


def test_residual_alignment_structure(model, tokens):
    result = mlp_residual_alignment(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_reinforcing"], bool)


def test_residual_alignment_cosine_range(model, tokens):
    result = mlp_residual_alignment(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.1 <= p["cosine"] <= 1.1  # small tolerance


def test_contribution_ratio_structure(model, tokens):
    result = mlp_contribution_ratio(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["mean_ratio"] >= 0


def test_contribution_ratio_positive(model, tokens):
    result = mlp_contribution_ratio(model, tokens)
    for p in result["per_layer"]:
        assert p["residual_norm"] >= 0
        assert p["mlp_norm"] >= 0


def test_decomposition_structure(model, tokens):
    result = mlp_residual_decomposition(model, tokens, layer=0, position=-1)
    assert result["parallel_norm"] >= 0
    assert result["perpendicular_norm"] >= 0
    assert result["total_norm"] >= 0


def test_decomposition_fractions_sum(model, tokens):
    result = mlp_residual_decomposition(model, tokens, layer=0, position=0)
    total = result["parallel_fraction"] ** 2 + result["perpendicular_fraction"] ** 2
    assert abs(total - 1.0) < 0.1  # approximately pythagorean


def test_attn_mlp_balance_structure(model, tokens):
    result = mlp_vs_attention_balance(model, tokens)
    assert len(result["per_layer"]) == 2
    total = result["attn_dominant_layers"] + result["mlp_dominant_layers"]
    assert total == 2


def test_attn_mlp_balance_fractions(model, tokens):
    result = mlp_vs_attention_balance(model, tokens)
    for p in result["per_layer"]:
        assert abs(p["attn_fraction"] + p["mlp_fraction"] - 1.0) < 0.01


def test_summary_structure(model, tokens):
    result = mlp_residual_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mlp_alignment" in p
        assert "mlp_norm" in p
        assert "attn_norm" in p
