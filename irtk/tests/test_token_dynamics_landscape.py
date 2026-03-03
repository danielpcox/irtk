"""Tests for token_dynamics_landscape module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_dynamics_landscape import (
    token_velocity, token_curvature, token_convergence,
    token_representation_drift, token_dynamics_landscape_summary,
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

def test_velocity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_velocity(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    assert "mean_velocity_per_layer" in result
    assert "fastest_layer" in result

def test_velocity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_velocity(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_velocity"] >= 0
        assert p["max_velocity"] >= p["mean_velocity"]
        assert len(p["per_position"]) == len(tokens)

def test_curvature_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_curvature(model, tokens)
    assert "per_transition" in result
    assert len(result["per_transition"]) == 1  # 2 layers -> 1 transition
    assert "mean_curvature" in result

def test_curvature_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_curvature(model, tokens)
    for p in result["per_transition"]:
        assert -1.0 <= p["mean_cosine"] <= 1.0
        assert p["curvature"] >= 0  # since curvature = 1 - cos

def test_convergence_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_convergence(model, tokens)
    assert "per_layer" in result
    assert "trend" in result
    assert result["trend"] in ("converging", "diverging", "stable")

def test_convergence_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_convergence(model, tokens)
    for p in result["per_layer"]:
        assert -1.0 <= p["mean_pairwise_similarity"] <= 1.0

def test_drift_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_representation_drift(model, tokens, reference_layer=0)
    assert "per_layer" in result
    assert result["reference_layer"] == 0
    assert len(result["per_layer"]) == 2

def test_drift_zero_at_reference(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_representation_drift(model, tokens, reference_layer=0)
    # Drift at reference layer should be ~0
    assert abs(result["per_layer"][0]["mean_drift"]) < 0.01

def test_dynamics_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = token_dynamics_landscape_summary(model, tokens)
    assert "fastest_layer" in result
    assert "mean_curvature" in result
    assert "convergence_trend" in result
    assert "per_layer" in result
