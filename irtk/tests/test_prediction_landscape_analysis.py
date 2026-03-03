"""Tests for prediction_landscape_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_landscape_analysis import (
    logit_distribution_profile, decision_margin_analysis,
    prediction_entropy_profile, logit_concentration,
    prediction_landscape_summary,
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


def test_distribution_profile_structure(model, tokens):
    result = logit_distribution_profile(model, tokens, position=-1)
    assert result["std"] >= 0
    assert result["entropy"] >= 0
    assert result["max"] >= result["min"]


def test_distribution_profile_range(model, tokens):
    result = logit_distribution_profile(model, tokens, position=-1)
    assert result["range"] == result["max"] - result["min"]
    assert 0 <= result["top_token"] < 50


def test_decision_margin_structure(model, tokens):
    result = decision_margin_analysis(model, tokens, position=-1)
    assert len(result["top_tokens"]) == 5
    assert len(result["top_logits"]) == 5
    assert isinstance(result["is_confident"], bool)


def test_decision_margin_ordering(model, tokens):
    result = decision_margin_analysis(model, tokens, position=-1)
    for i in range(len(result["top_logits"]) - 1):
        assert result["top_logits"][i] >= result["top_logits"][i + 1]


def test_entropy_profile_structure(model, tokens):
    result = prediction_entropy_profile(model, tokens)
    assert len(result["per_position"]) == 5
    assert result["mean_entropy"] >= 0


def test_entropy_profile_values(model, tokens):
    result = prediction_entropy_profile(model, tokens)
    for p in result["per_position"]:
        assert p["entropy"] >= 0
        assert 0 <= p["top_prob"] <= 1


def test_concentration_structure(model, tokens):
    result = logit_concentration(model, tokens, position=-1)
    assert 0 <= result["top_1_prob"] <= 1
    assert result["top_5_prob"] >= result["top_1_prob"]
    assert result["top_10_prob"] >= result["top_5_prob"]


def test_summary_structure(model, tokens):
    result = prediction_landscape_summary(model, tokens, position=-1)
    assert result["entropy"] >= 0
    assert isinstance(result["is_confident"], bool)


def test_summary_consistency(model, tokens):
    result = prediction_landscape_summary(model, tokens, position=-1)
    assert 0 <= result["top_1_prob"] <= 1
    assert result["effective_tokens"] >= 1
    assert result["logit_std"] >= 0
