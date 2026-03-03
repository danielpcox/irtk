"""Tests for layer_skip_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_skip_analysis import (
    layer_skip_logit_impact, layer_residual_contribution,
    layer_redundancy_analysis, layer_cumulative_effect,
    layer_skip_summary,
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


def test_skip_logit_impact_structure(model, tokens):
    result = layer_skip_logit_impact(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["most_critical_layer"] < 2


def test_skip_logit_impact_mse(model, tokens):
    result = layer_skip_logit_impact(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["mse_logit_change"] >= 0


def test_residual_contribution_structure(model, tokens):
    result = layer_residual_contribution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["mean_contribution"] >= 0


def test_residual_contribution_norms(model, tokens):
    result = layer_residual_contribution(model, tokens)
    for p in result["per_layer"]:
        assert p["contribution_norm"] >= 0
        assert p["residual_norm"] >= 0


def test_redundancy_analysis_structure(model, tokens):
    result = layer_redundancy_analysis(model, tokens)
    assert len(result["pairs"]) == 1  # 2 layers = 1 pair


def test_redundancy_analysis_similarity(model, tokens):
    result = layer_redundancy_analysis(model, tokens)
    for p in result["pairs"]:
        assert -1 <= p["similarity"] <= 1


def test_cumulative_effect_structure(model, tokens):
    result = layer_cumulative_effect(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["n_changes"] >= 0


def test_cumulative_effect_entropy(model, tokens):
    result = layer_cumulative_effect(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0


def test_summary_structure(model, tokens):
    result = layer_skip_summary(model, tokens, position=-1)
    assert 0 <= result["most_critical_layer"] < 2
    assert len(result["per_layer"]) == 2
