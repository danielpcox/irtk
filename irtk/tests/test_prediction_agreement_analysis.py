"""Tests for prediction_agreement_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_agreement_analysis import (
    logit_lens_agreement, head_prediction_agreement,
    component_prediction_comparison, prediction_stability,
    prediction_agreement_summary,
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


def test_logit_lens_agreement_structure(model, tokens):
    result = logit_lens_agreement(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["agreement_fraction"] <= 1


def test_logit_lens_commit_layer(model, tokens):
    result = logit_lens_agreement(model, tokens, position=-1)
    assert 0 <= result["commit_layer"] < 2


def test_head_prediction_agreement_structure(model, tokens):
    result = head_prediction_agreement(model, tokens, layer=-1, position=-1)
    assert len(result["per_head"]) == 4
    assert 0 <= result["agreement_fraction"] <= 1


def test_head_prediction_unique(model, tokens):
    result = head_prediction_agreement(model, tokens, layer=0, position=-1)
    assert 1 <= result["n_unique_predictions"] <= 4


def test_component_comparison_structure(model, tokens):
    result = component_prediction_comparison(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["agreement_fraction"] <= 1


def test_component_comparison_correlation(model, tokens):
    result = component_prediction_comparison(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert -1.1 <= p["logit_correlation"] <= 1.1


def test_prediction_stability_structure(model, tokens):
    result = prediction_stability(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_stable"], bool)


def test_prediction_stability_changes(model, tokens):
    result = prediction_stability(model, tokens, position=-1)
    assert result["total_changes"] >= 0


def test_summary_structure(model, tokens):
    result = prediction_agreement_summary(model, tokens, position=-1)
    assert "final_prediction" in result
    assert "commit_layer" in result
    assert isinstance(result["is_stable"], bool)
    assert 0 <= result["layer_agreement"] <= 1
