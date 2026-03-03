"""Tests for embedding_composition_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_composition_analysis import (
    token_position_balance, token_position_alignment,
    combined_embedding_properties, embedding_subspace_analysis,
    embedding_composition_summary,
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


def test_balance_structure(model, tokens):
    result = token_position_balance(model, tokens)
    assert len(result["per_position"]) == 5
    assert "dominant" in result


def test_balance_fractions(model, tokens):
    result = token_position_balance(model, tokens)
    for p in result["per_position"]:
        assert 0 <= p["token_fraction"] <= 1
        assert p["token_norm"] >= 0
        assert p["position_norm"] >= 0


def test_alignment_structure(model, tokens):
    result = token_position_alignment(model, tokens)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_aligned"], bool)


def test_alignment_cosine_range(model, tokens):
    result = token_position_alignment(model, tokens)
    for p in result["per_position"]:
        assert -1.1 <= p["cosine"] <= 1.1


def test_combined_properties_structure(model, tokens):
    result = combined_embedding_properties(model, tokens)
    assert len(result["per_position"]) == 5
    assert result["mean_norm"] > 0


def test_combined_similarity_range(model, tokens):
    result = combined_embedding_properties(model, tokens)
    assert -1.1 <= result["mean_pairwise_similarity"] <= 1.1


def test_subspace_analysis(model, tokens):
    result = embedding_subspace_analysis(model, tokens)
    assert result["effective_rank"] > 0
    assert result["dim_for_90_pct"] >= 1
    assert result["n_tokens"] == 5


def test_summary_structure(model, tokens):
    result = embedding_composition_summary(model, tokens)
    assert "token_fraction" in result
    assert "dominant" in result
    assert "effective_rank" in result


def test_summary_consistency(model, tokens):
    result = embedding_composition_summary(model, tokens)
    assert result["dominant"] in ("token", "position")
    assert result["effective_rank"] > 0
    assert result["mean_norm"] > 0
