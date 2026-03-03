"""Tests for training_signal_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.training_signal_analysis import (
    per_token_loss_analysis, loss_component_attribution,
    prediction_entropy_profile, loss_concentration_analysis,
    training_signal_summary,
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


def test_per_token_loss_structure(model, tokens):
    result = per_token_loss_analysis(model, tokens)
    assert len(result["per_position"]) == 4  # seq_len - 1
    assert result["mean_loss"] >= 0


def test_per_token_loss_values(model, tokens):
    result = per_token_loss_analysis(model, tokens)
    for p in result["per_position"]:
        assert p["loss"] >= 0
        assert 0 <= p["probability"] <= 1
        assert p["rank"] >= 0


def test_loss_component_attribution_structure(model, tokens):
    result = loss_component_attribution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["total_attn"], float)


def test_loss_component_helps(model, tokens):
    result = loss_component_attribution(model, tokens)
    for p in result["per_layer"]:
        assert isinstance(p["helps_prediction"], bool)


def test_prediction_entropy_structure(model, tokens):
    result = prediction_entropy_profile(model, tokens)
    assert len(result["per_position"]) == 5
    assert result["mean_entropy"] >= 0


def test_prediction_entropy_range(model, tokens):
    result = prediction_entropy_profile(model, tokens)
    for p in result["per_position"]:
        assert p["entropy"] >= 0
        assert 0 <= p["max_prob"] <= 1


def test_loss_concentration_structure(model, tokens):
    result = loss_concentration_analysis(model, tokens)
    assert 0 <= result["top_20pct_loss_fraction"] <= 1
    assert isinstance(result["is_concentrated"], bool)


def test_loss_concentration_gini(model, tokens):
    result = loss_concentration_analysis(model, tokens)
    assert -1 <= result["gini"] <= 1


def test_training_signal_summary_structure(model, tokens):
    result = training_signal_summary(model, tokens)
    assert len(result["per_position"]) > 0
    assert result["mean_loss"] >= 0
    assert result["mean_entropy"] >= 0
