"""Tests for residual direction tracking."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_direction_tracking import (
    direction_persistence, direction_change_rate,
    unembed_direction_trajectory, dominant_direction_evolution,
    residual_direction_tracking_summary,
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


def test_direction_persistence_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = direction_persistence(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_direction_persistence_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = direction_persistence(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert "mean_persistence" in p
        assert "min_persistence" in p


def test_direction_change_rate_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = direction_change_rate(model, tokens, position=-1)
    assert "per_transition" in result
    assert "mean_angle" in result
    assert "max_angle" in result
    assert len(result["per_transition"]) == 1  # 2 layers -> 1 transition


def test_direction_change_rate_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = direction_change_rate(model, tokens, position=-1)
    for t in result["per_transition"]:
        assert 0 <= t["angle_degrees"] <= 180
        assert -1.0 <= t["cosine"] <= 1.0


def test_unembed_trajectory_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = unembed_direction_trajectory(model, tokens, position=-1, token_id=5)
    assert "per_layer" in result
    assert "token_id" in result
    assert result["token_id"] == 5
    assert len(result["per_layer"]) == 2


def test_unembed_trajectory_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = unembed_direction_trajectory(model, tokens, position=-1, token_id=5)
    for p in result["per_layer"]:
        assert "cosine" in p
        assert "projection" in p


def test_dominant_direction_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = dominant_direction_evolution(model, tokens, position=-1)
    assert "per_layer" in result
    assert "n_changes" in result
    assert "final_token" in result
    assert len(result["per_layer"]) == 2


def test_dominant_direction_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = dominant_direction_evolution(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert "top_token" in p
        assert "top_logit" in p
        assert "changed" in p
    assert result["per_layer"][0]["changed"] == False  # First layer can't change


def test_direction_tracking_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_direction_tracking_summary(model, tokens, position=-1)
    assert "mean_change_angle" in result
    assert "max_change_angle" in result
    assert "n_prediction_changes" in result
    assert "final_token" in result
