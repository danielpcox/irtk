"""Tests for token prediction dynamics."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_prediction_dynamics import (
    prediction_trajectory, prediction_confidence_evolution,
    prediction_rank_tracking, prediction_stability,
    token_prediction_dynamics_summary,
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


def test_prediction_trajectory_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_trajectory(model, tokens, position=-1)
    assert "per_layer" in result
    assert "n_changes" in result
    assert len(result["per_layer"]) == 2


def test_prediction_trajectory_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_trajectory(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert 0 <= p["predicted_token"] < 50
        assert 0 < p["predicted_prob"] <= 1.0
    assert result["per_layer"][0]["changed_from_previous"] is False


def test_confidence_evolution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_confidence_evolution(model, tokens, position=-1)
    assert "per_layer" in result
    assert "final_entropy" in result
    assert len(result["per_layer"]) == 2


def test_confidence_evolution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_confidence_evolution(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
        assert 0 < p["max_prob"] <= 1.0


def test_rank_tracking_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_rank_tracking(model, tokens, position=-1)
    assert "tracked_tokens" in result
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_rank_tracking_explicit_tokens(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_rank_tracking(model, tokens, position=-1, track_tokens=[1, 5])
    assert result["tracked_tokens"] == [1, 5]
    for p in result["per_layer"]:
        assert 1 in p["ranks"]
        assert 5 in p["ranks"]
        assert 0 <= p["ranks"][1] < 50


def test_stability_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_stability(model, tokens, position=-1)
    assert "per_transition" in result
    assert "mean_stability" in result
    assert len(result["per_transition"]) == 1  # 2 layers -> 1 transition


def test_stability_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_stability(model, tokens, position=-1)
    for p in result["per_transition"]:
        assert -1.0 <= p["cosine_similarity"] <= 1.0


def test_dynamics_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_prediction_dynamics_summary(model, tokens, position=-1)
    assert "n_changes" in result
    assert "final_entropy" in result
    assert "mean_stability" in result
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
