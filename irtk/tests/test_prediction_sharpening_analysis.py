"""Tests for prediction sharpening analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_sharpening_analysis import (
    sharpening_trajectory, component_sharpening_contribution,
    top_k_probability_evolution, sharpening_rate,
    prediction_sharpening_summary,
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


def test_sharpening_trajectory_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = sharpening_trajectory(model, tokens, position=-1)
    assert "per_layer" in result
    assert "initial_entropy" in result
    assert "final_entropy" in result
    assert "total_sharpening" in result
    assert len(result["per_layer"]) == 2


def test_sharpening_trajectory_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = sharpening_trajectory(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0
        assert 0 <= p["top_prob"] <= 1.0
    assert abs(result["total_sharpening"] - (result["initial_entropy"] - result["final_entropy"])) < 1e-4


def test_component_sharpening_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_sharpening_contribution(model, tokens, layer=0, position=-1)
    assert "entropy_pre" in result
    assert "entropy_mid" in result
    assert "entropy_post" in result
    assert "attn_sharpening" in result
    assert "mlp_sharpening" in result


def test_component_sharpening_consistency(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_sharpening_contribution(model, tokens, layer=0, position=-1)
    total = result["attn_sharpening"] + result["mlp_sharpening"]
    assert abs(total - result["total_sharpening"]) < 1e-4


def test_top_k_evolution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = top_k_probability_evolution(model, tokens, position=-1, top_k=3)
    assert "tracked_tokens" in result
    assert "per_layer" in result
    assert len(result["tracked_tokens"]) == 3
    assert len(result["per_layer"]) == 2


def test_top_k_evolution_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = top_k_probability_evolution(model, tokens, position=-1, top_k=3)
    for p in result["per_layer"]:
        assert "token_probs" in p
        assert "top_k_mass" in p
        assert 0 <= p["top_k_mass"] <= 1.0


def test_sharpening_rate_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = sharpening_rate(model, tokens, position=-1)
    assert "per_transition" in result
    assert "fastest_sharpening_layer" in result
    assert "max_rate" in result
    assert len(result["per_transition"]) == 1  # 2 layers -> 1 transition


def test_sharpening_rate_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = sharpening_rate(model, tokens, position=-1)
    for t in result["per_transition"]:
        assert "rate" in t
        assert "is_sharpening" in t


def test_prediction_sharpening_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = prediction_sharpening_summary(model, tokens, position=-1)
    assert "total_sharpening" in result
    assert "initial_entropy" in result
    assert "final_entropy" in result
    assert "fastest_sharpening_layer" in result
    assert "max_rate" in result
