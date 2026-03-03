"""Tests for attention residual decomposition."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_residual_decomposition import (
    attention_parallel_perpendicular, per_head_residual_decomposition,
    attention_update_angle, attention_residual_ratio,
    attention_residual_decomposition_summary,
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


def test_parallel_perpendicular_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_parallel_perpendicular(model, tokens, layer=0, position=-1)
    assert "parallel_magnitude" in result
    assert "perpendicular_magnitude" in result
    assert "total_magnitude" in result
    assert "parallel_fraction" in result
    assert "is_reinforcing" in result


def test_parallel_perpendicular_pythagorean(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_parallel_perpendicular(model, tokens, layer=0, position=-1)
    recon = result["parallel_magnitude"] ** 2 + result["perpendicular_magnitude"] ** 2
    assert abs(recon - result["total_magnitude"] ** 2) < 0.01


def test_per_head_decomposition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_head_residual_decomposition(model, tokens, layer=0, position=-1)
    assert "per_head" in result
    assert len(result["per_head"]) == 4
    assert "n_reinforcing" in result


def test_per_head_decomposition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = per_head_residual_decomposition(model, tokens, layer=0, position=-1)
    for h in result["per_head"]:
        assert "parallel" in h
        assert "perpendicular" in h
        assert h["perpendicular"] >= 0
        assert h["total_norm"] >= 0


def test_update_angle_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_update_angle(model, tokens, layer=0)
    assert "per_position" in result
    assert "mean_angle" in result
    assert len(result["per_position"]) == 5


def test_update_angle_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_update_angle(model, tokens, layer=0)
    for p in result["per_position"]:
        assert 0 <= p["angle_degrees"] <= 180
        assert -1.0 <= p["cosine"] <= 1.0


def test_residual_ratio_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_residual_ratio(model, tokens, layer=0)
    assert "per_position" in result
    assert "mean_ratio" in result
    assert len(result["per_position"]) == 5


def test_residual_ratio_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_residual_ratio(model, tokens, layer=0)
    for p in result["per_position"]:
        assert p["residual_norm"] >= 0
        assert p["attention_norm"] >= 0
        assert p["ratio"] >= 0


def test_decomposition_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_residual_decomposition_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "parallel_fraction" in p
        assert "mean_angle" in p
