"""Tests for attention flow analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_flow_analysis import (
    attention_rollout, source_token_attribution,
    layer_attention_entropy, attention_distance_profile,
    attention_flow_summary,
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


def test_rollout_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_rollout(model, tokens)
    assert "rollout_matrix" in result
    assert result["rollout_matrix"].shape == (5, 5)


def test_rollout_row_sums(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_rollout(model, tokens)
    row_sums = jnp.sum(result["rollout_matrix"], axis=-1)
    for s in row_sums:
        assert abs(float(s) - 1.0) < 0.1


def test_source_attribution_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_token_attribution(model, tokens, target_position=-1)
    assert "per_position" in result
    assert "most_influential" in result
    assert len(result["per_position"]) == 5


def test_source_attribution_sums(model_and_tokens):
    model, tokens = model_and_tokens
    result = source_token_attribution(model, tokens, target_position=-1)
    total = sum(p["attribution"] for p in result["per_position"])
    assert abs(total - 1.0) < 0.1


def test_entropy_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_attention_entropy(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_entropy_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_attention_entropy(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_entropy"] >= 0
        assert p["min_entropy"] <= p["max_entropy"]


def test_distance_profile_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_distance_profile(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4


def test_distance_profile_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_distance_profile(model, tokens, layer=0)
    for p in result["per_head"]:
        assert p["mean_distance"] >= 0


def test_flow_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_flow_summary(model, tokens)
    assert "per_layer" in result
    assert "rollout_shape" in result
    assert result["rollout_shape"] == [5, 5]
