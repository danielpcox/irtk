"""Tests for logit_lens_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_lens_decomposition import (
    per_layer_logit_lens, component_logit_contribution,
    prediction_change_attribution, logit_lens_entropy_trajectory,
    logit_lens_decomposition_summary,
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


def test_per_layer_logit_lens_structure(model, tokens):
    result = per_layer_logit_lens(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_per_layer_logit_lens_values(model, tokens):
    result = per_layer_logit_lens(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert 0 <= p["top_prob"] <= 1
        assert p["entropy"] >= 0
        assert len(p["top_k_tokens"]) == 5


def test_component_contribution_structure(model, tokens):
    result = component_logit_contribution(model, tokens, layer=0, position=-1)
    assert len(result["per_token"]) == 5
    assert abs(result["attn_fraction"] + result["mlp_fraction"] - 1.0) < 0.01


def test_component_contribution_values(model, tokens):
    result = component_logit_contribution(model, tokens, layer=0, position=-1)
    for p in result["per_token"]:
        assert "attn_logit" in p
        assert "mlp_logit" in p


def test_prediction_change_structure(model, tokens):
    result = prediction_change_attribution(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["n_prediction_changes"] >= 0


def test_prediction_change_values(model, tokens):
    result = prediction_change_attribution(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert isinstance(p["changed_prediction"], bool)
        assert 0 <= p["top_token"] < 50


def test_entropy_trajectory_structure(model, tokens):
    result = logit_lens_entropy_trajectory(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_sharpening"], bool)


def test_summary_structure(model, tokens):
    result = logit_lens_decomposition_summary(model, tokens, position=-1)
    assert result["n_prediction_changes"] >= 0
    assert isinstance(result["is_sharpening"], bool)


def test_summary_values(model, tokens):
    result = logit_lens_decomposition_summary(model, tokens, position=-1)
    assert 0 <= result["final_token"] < 50
