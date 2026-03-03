"""Tests for attention_head_pruning module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_pruning import (
    head_ablation_impact, head_importance_ranking,
    head_pruning_tolerance, head_output_norm_distribution,
    head_pruning_summary,
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


def test_ablation_impact_structure(model, tokens):
    result = head_ablation_impact(model, tokens, layer=0, position=-1)
    assert len(result["per_head"]) == 4
    assert 0 <= result["most_impactful_head"] < 4


def test_ablation_impact_mse(model, tokens):
    result = head_ablation_impact(model, tokens, layer=0, position=-1)
    for h in result["per_head"]:
        assert h["mse_change"] >= 0


def test_importance_ranking_structure(model, tokens):
    result = head_importance_ranking(model, tokens, position=-1)
    assert len(result["ranking"]) == 8  # 2 layers * 4 heads


def test_importance_ranking_sorted(model, tokens):
    result = head_importance_ranking(model, tokens, position=-1)
    importances = [h["importance"] for h in result["ranking"]]
    assert importances == sorted(importances, reverse=True)


def test_pruning_tolerance_structure(model, tokens):
    result = head_pruning_tolerance(model, tokens, position=-1)
    assert result["total_heads"] == 8
    assert 0 <= result["heads_prunable"] <= 8
    assert 0 <= result["pruning_tolerance"] <= 1


def test_output_norm_distribution(model, tokens):
    result = head_output_norm_distribution(model, tokens, position=-1)
    assert len(result["per_head"]) == 8
    assert result["mean_norm"] >= 0
    assert result["max_norm"] >= result["min_norm"]


def test_output_norm_near_dead(model, tokens):
    result = head_output_norm_distribution(model, tokens, position=-1)
    assert result["n_near_dead"] >= 0


def test_summary_structure(model, tokens):
    result = head_pruning_summary(model, tokens, position=-1)
    assert "most_important" in result
    assert "least_important" in result
    assert 0 <= result["pruning_tolerance"] <= 1


def test_summary_consistency(model, tokens):
    result = head_pruning_summary(model, tokens, position=-1)
    assert result["heads_prunable"] <= result["total_heads"]
