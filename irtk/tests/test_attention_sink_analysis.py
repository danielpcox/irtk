"""Tests for attention sink analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_sink_analysis import (
    attention_sink_detection, bos_attention_profile,
    sink_formation_trajectory, attention_concentration,
    attention_sink_summary,
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


def test_sink_detection_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_sink_detection(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_sinks" in result
    assert len(result["per_head"]) == 4


def test_sink_detection_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_sink_detection(model, tokens, layer=0, threshold=0.3)
    assert result["n_sinks"] >= 0
    for h in result["per_head"]:
        for s in h["sinks"]:
            assert s["mean_attention_received"] > 0.3


def test_bos_profile_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = bos_attention_profile(model, tokens, position=0)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_bos_profile_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = bos_attention_profile(model, tokens, position=0)
    for p in result["per_layer"]:
        assert 0 <= p["mean_bos_attention"] <= 1.0
        for h in p["per_head"]:
            assert 0 <= h["bos_attention"] <= 1.0


def test_sink_formation_trajectory(model_and_tokens):
    model, tokens = model_and_tokens
    result = sink_formation_trajectory(model, tokens, position=0)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["mean_attention"] >= 0
        assert p["max_attention"] >= p["mean_attention"]


def test_concentration_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_concentration(model, tokens, layer=0)
    assert "per_head" in result
    assert "mean_gini" in result
    assert len(result["per_head"]) == 4


def test_concentration_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_concentration(model, tokens, layer=0)
    for p in result["per_head"]:
        assert -1.0 <= p["gini"] <= 1.0


def test_sink_concentration_has_gini(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_concentration(model, tokens, layer=0)
    assert isinstance(result["mean_gini"], float)


def test_sink_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_sink_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "n_sinks" in p
        assert "mean_gini" in p
