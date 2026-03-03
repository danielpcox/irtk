"""Tests for MLP gating analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_gating_analysis import (
    mlp_activation_sparsity, mlp_pre_post_correlation,
    mlp_activation_distribution, mlp_gating_selectivity,
    mlp_gating_summary,
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


def test_sparsity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_sparsity(model, tokens, layer=0)
    assert "sparsity" in result
    assert "per_position" in result
    assert len(result["per_position"]) == 5


def test_sparsity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_sparsity(model, tokens, layer=0)
    assert 0 <= result["sparsity"] <= 1.0
    for p in result["per_position"]:
        assert 0 <= p["sparsity"] <= 1.0


def test_pre_post_correlation(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_pre_post_correlation(model, tokens, layer=0)
    assert "mean_correlation" in result
    assert len(result["per_position"]) == 5


def test_pre_post_correlation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_pre_post_correlation(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.0 <= p["correlation"] <= 1.0


def test_distribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_distribution(model, tokens, layer=0)
    assert "mean" in result
    assert "std" in result
    assert "skewness" in result
    assert "kurtosis" in result


def test_distribution_std_positive(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_activation_distribution(model, tokens, layer=0)
    assert result["std"] >= 0
    assert result["max"] >= result["min"]


def test_gating_selectivity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gating_selectivity(model, tokens, layer=0, top_k=3)
    assert "most_selective" in result
    assert "least_selective" in result
    assert len(result["most_selective"]) == 3


def test_gating_selectivity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gating_selectivity(model, tokens, layer=0, top_k=3)
    for _, sel in result["most_selective"]:
        assert 0 <= sel <= 1.0
    assert 0 <= result["mean_selectivity"] <= 1.0


def test_gating_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_gating_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "sparsity" in p
        assert "skewness" in p
