"""Tests for MLP residual alignment."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_residual_alignment import (
    mlp_residual_cosine, mlp_parallel_perpendicular,
    mlp_contribution_ratio, mlp_unembed_alignment,
    mlp_residual_alignment_summary,
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


def test_residual_cosine_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_cosine(model, tokens, layer=0)
    assert "per_position" in result
    assert "mean_cosine" in result
    assert "n_reinforcing" in result
    assert len(result["per_position"]) == 5


def test_residual_cosine_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_cosine(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.0 <= p["cosine"] <= 1.0


def test_parallel_perpendicular_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_parallel_perpendicular(model, tokens, layer=0, position=-1)
    assert "parallel_magnitude" in result
    assert "perpendicular_magnitude" in result
    assert "parallel_fraction" in result


def test_parallel_perpendicular_pythagorean(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_parallel_perpendicular(model, tokens, layer=0, position=-1)
    recon = result["parallel_magnitude"] ** 2 + result["perpendicular_magnitude"] ** 2
    assert abs(recon - result["total_magnitude"] ** 2) < 0.01


def test_contribution_ratio_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_ratio(model, tokens, layer=0)
    assert "per_position" in result
    assert "mean_ratio" in result
    assert len(result["per_position"]) == 5


def test_contribution_ratio_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_contribution_ratio(model, tokens, layer=0)
    for p in result["per_position"]:
        assert p["ratio"] >= 0


def test_unembed_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_unembed_alignment(model, tokens, layer=0, position=-1, top_k=5)
    assert "promoted" in result
    assert "suppressed" in result
    assert len(result["promoted"]) == 5


def test_unembed_alignment_ordering(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_unembed_alignment(model, tokens, layer=0, position=-1, top_k=5)
    promoted_logits = [l for _, l in result["promoted"]]
    assert promoted_logits == sorted(promoted_logits, reverse=True)


def test_alignment_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_residual_alignment_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_cosine" in p
        assert "mean_ratio" in p
