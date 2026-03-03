"""Tests for token_context_buildup module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_context_buildup import (
    context_accumulation_rate, context_source_attribution,
    position_context_diversity, embedding_distance_tracking,
    context_buildup_summary,
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


def test_accumulation_rate_structure(model, tokens):
    result = context_accumulation_rate(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["update_trend"] in ("increasing", "decreasing", "stable")


def test_accumulation_rate_positive(model, tokens):
    result = context_accumulation_rate(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["update_norm"] >= 0
        assert p["relative_update"] >= 0


def test_source_attribution_structure(model, tokens):
    result = context_source_attribution(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["mean_attn_fraction"] <= 1


def test_source_attribution_fractions(model, tokens):
    result = context_source_attribution(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert abs(p["attn_fraction"] + p["mlp_fraction"] - 1.0) < 0.01


def test_position_diversity_structure(model, tokens):
    result = position_context_diversity(model, tokens, layer=-1)
    assert -1 <= result["mean_pairwise_similarity"] <= 1
    assert isinstance(result["is_diverse"], bool)


def test_embedding_distance_structure(model, tokens):
    result = embedding_distance_tracking(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["final_distance"] >= 0


def test_embedding_distance_cosine(model, tokens):
    result = embedding_distance_tracking(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert -1.1 <= p["cosine_to_embed"] <= 1.1


def test_summary_structure(model, tokens):
    result = context_buildup_summary(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["update_trend"] in ("increasing", "decreasing", "stable")


def test_summary_fields(model, tokens):
    result = context_buildup_summary(model, tokens, position=-1)
    assert result["final_distance"] >= 0
    assert -1.1 <= result["final_cosine"] <= 1.1
    assert 0 <= result["mean_attn_fraction"] <= 1
