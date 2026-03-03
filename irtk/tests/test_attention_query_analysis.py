"""Tests for attention_query_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_query_analysis import (
    query_norm_profile, query_diversity,
    query_key_matching, query_subspace_analysis,
    query_analysis_summary,
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


def test_norm_profile_structure(model, tokens):
    result = query_norm_profile(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["mean_query_norm"] > 0


def test_norm_profile_values(model, tokens):
    result = query_norm_profile(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_norm"] > 0
        assert h["max_norm"] >= h["mean_norm"]
        assert h["std_norm"] >= 0


def test_diversity_structure(model, tokens):
    result = query_diversity(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert 0 <= result["mean_diversity"] <= 1


def test_diversity_similarity_range(model, tokens):
    result = query_diversity(model, tokens, layer=0)
    for h in result["per_head"]:
        assert -1.1 <= h["mean_query_similarity"] <= 1.1
        assert isinstance(h["is_diverse"], bool)


def test_key_matching_structure(model, tokens):
    result = query_key_matching(model, tokens, layer=0, head=0)
    assert "mean_score" in result
    assert "score_range" in result
    assert result["score_range"] >= 0


def test_key_matching_stats(model, tokens):
    result = query_key_matching(model, tokens, layer=0, head=0)
    assert result["max_score"] >= result["min_score"]
    assert result["std_score"] >= 0


def test_subspace_analysis(model, tokens):
    result = query_subspace_analysis(model, tokens, layer=0, head=0)
    assert result["effective_rank"] > 0
    assert result["top_sv"] > 0
    assert result["sv_ratio"] >= 1


def test_summary_structure(model, tokens):
    result = query_analysis_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = query_analysis_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_query_norm"] > 0
        assert 0 <= p["diversity_fraction"] <= 1
