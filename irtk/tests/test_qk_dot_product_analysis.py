"""Tests for QK dot product analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.qk_dot_product_analysis import (
    qk_score_statistics, qk_temperature_analysis,
    qk_positional_bias, qk_content_vs_position,
    qk_dot_product_summary,
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


def test_score_statistics_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_score_statistics(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4


def test_score_statistics_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_score_statistics(model, tokens, layer=0)
    for h in result["per_head"]:
        assert "mean_score" in h
        assert "std_score" in h
        assert h["max_score"] >= h["min_score"]


def test_temperature_analysis_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_temperature_analysis(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_sharp" in result
    assert len(result["per_head"]) == 4


def test_temperature_analysis_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_temperature_analysis(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_score_range"] >= 0


def test_positional_bias_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_positional_bias(model, tokens, layer=0, head=0)
    assert "per_distance" in result
    assert len(result["per_distance"]) > 0


def test_positional_bias_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_positional_bias(model, tokens, layer=0, head=0)
    for d in result["per_distance"]:
        assert d["distance"] >= 0
        assert d["n_samples"] > 0


def test_content_vs_position_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_content_vs_position(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_positional" in result
    assert len(result["per_head"]) == 4


def test_content_vs_position_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_content_vs_position(model, tokens, layer=0)
    for h in result["per_head"]:
        assert 0 <= h["position_fraction"] <= 1.0
        assert abs(h["position_fraction"] + h["content_fraction"] - 1.0) < 1e-6


def test_dot_product_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_dot_product_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
