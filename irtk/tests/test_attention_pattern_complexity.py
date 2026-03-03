"""Tests for attention_pattern_complexity module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_complexity import (
    pattern_entropy_complexity, pattern_rank_complexity,
    pattern_regularity, pattern_stability_across_positions,
    pattern_complexity_summary,
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


def test_entropy_complexity_structure(model, tokens):
    result = pattern_entropy_complexity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["n_complex_heads"] >= 0


def test_entropy_complexity_values(model, tokens):
    result = pattern_entropy_complexity(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_entropy"] >= 0
        assert 0 <= h["normalized_entropy"] <= 1.1
        assert isinstance(h["is_complex"], bool)


def test_rank_complexity_structure(model, tokens):
    result = pattern_rank_complexity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["mean_rank"] > 0


def test_rank_complexity_values(model, tokens):
    result = pattern_rank_complexity(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["effective_rank"] > 0
        assert h["top_sv"] > 0


def test_regularity_structure(model, tokens):
    result = pattern_regularity(model, tokens, layer=0, head=0)
    assert 0 <= result["self_attention"] <= 1
    assert 0 <= result["first_token_attention"] <= 1
    assert result["dominant_pattern"] in ("self", "prev", "first")


def test_regularity_values(model, tokens):
    result = pattern_regularity(model, tokens, layer=0, head=0)
    assert 0 <= result["prev_token_attention"] <= 1


def test_stability_structure(model, tokens):
    result = pattern_stability_across_positions(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert 0 <= result["mean_stability"] <= 1


def test_summary_structure(model, tokens):
    result = pattern_complexity_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = pattern_complexity_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["n_complex_heads"] >= 0
        assert p["mean_rank"] > 0
