"""Tests for feature_emergence_tracking module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.feature_emergence_tracking import (
    feature_probe_trajectory, token_feature_emergence,
    component_feature_contribution, feature_interference,
    feature_emergence_summary,
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


def test_probe_trajectory_structure(model, tokens):
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_probe_trajectory(model, tokens, direction)
    assert len(result["per_layer"]) == 2
    assert result["max_projection"] >= 0


def test_probe_trajectory_emergence(model, tokens):
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = feature_probe_trajectory(model, tokens, direction)
    assert 0 <= result["emergence_layer"] < 2


def test_token_feature_emergence_structure(model, tokens):
    result = token_feature_emergence(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["commit_layer"] < 2


def test_token_feature_emergence_probs(model, tokens):
    result = token_feature_emergence(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["top_prob"] <= 1
        assert p["entropy"] >= 0


def test_component_contribution_structure(model, tokens):
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = component_feature_contribution(model, tokens, direction)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["dominant"] in ("attention", "mlp")


def test_feature_interference_structure(model, tokens):
    d1 = jax.random.normal(jax.random.PRNGKey(0), (16,))
    d2 = jax.random.normal(jax.random.PRNGKey(1), (16,))
    result = feature_interference(model, tokens, [d1, d2])
    assert result["n_directions"] == 2
    assert isinstance(result["high_interference"], bool)


def test_feature_interference_pairs(model, tokens):
    d1 = jax.random.normal(jax.random.PRNGKey(0), (16,))
    d2 = jax.random.normal(jax.random.PRNGKey(1), (16,))
    result = feature_interference(model, tokens, [d1, d2])
    for p in result["per_layer"]:
        assert len(p["projections"]) == 2
        assert len(p["direction_pairs"]) == 1


def test_emergence_summary_structure(model, tokens):
    result = feature_emergence_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
        assert 0 <= p["max_prob"] <= 1


def test_emergence_summary_entropy(model, tokens):
    result = feature_emergence_summary(model, tokens)
    # Entropy reduction can be positive or negative for random weights
    assert isinstance(result["entropy_reduction"], float)
