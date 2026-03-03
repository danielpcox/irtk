"""Tests for head writing direction analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_writing_direction_analysis import (
    head_writing_directions, head_unembed_alignment,
    head_direction_consistency, head_writing_magnitude,
    head_writing_summary,
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


def test_head_writing_directions_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_directions(model, tokens, layer=0)
    assert "layer" in result
    assert "per_head" in result
    assert "mean_rank" in result
    assert len(result["per_head"]) == 4


def test_head_writing_directions_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_directions(model, tokens, layer=0)
    for h in result["per_head"]:
        assert "effective_rank" in h
        assert "top_sv" in h
        assert h["effective_rank"] >= 1.0
        assert h["top_sv"] > 0
        assert 0 <= h["sv_concentration"] <= 1.0


def test_head_unembed_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_unembed_alignment(model, tokens, layer=0, top_k=5)
    assert "layer" in result
    assert "per_head" in result
    assert len(result["per_head"]) == 4
    for h in result["per_head"]:
        assert len(h["top_tokens"]) == 5
        assert len(h["top_logits"]) == 5


def test_head_direction_consistency_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_direction_consistency(model, tokens, layer=0)
    assert "per_head" in result
    assert "n_consistent" in result
    for h in result["per_head"]:
        assert "direction_consistency" in h
        assert "is_consistent" in h


def test_head_direction_consistency_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_direction_consistency(model, tokens, layer=0)
    for h in result["per_head"]:
        assert -1.0 <= h["direction_consistency"] <= 1.0


def test_head_writing_magnitude_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_magnitude(model, tokens, layer=0)
    assert "per_head" in result
    assert "dominant_head" in result
    assert len(result["per_head"]) == 4


def test_head_writing_magnitude_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_magnitude(model, tokens, layer=0)
    total = sum(h["fraction"] for h in result["per_head"])
    assert abs(total - 1.0) < 1e-4


def test_head_writing_summary_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2


def test_head_writing_summary_content(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_writing_summary(model, tokens)
    for p in result["per_layer"]:
        assert "layer" in p
        assert "mean_rank" in p
        assert "n_consistent" in p
        assert p["mean_rank"] >= 1.0
