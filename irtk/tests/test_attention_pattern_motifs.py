"""Tests for attention_pattern_motifs module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_motifs import (
    detect_diagonal_motif, detect_stripe_motif,
    detect_block_motif, detect_triangular_motif,
    attention_motif_summary,
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


def test_diagonal_motif_structure(model, tokens):
    result = detect_diagonal_motif(model, tokens, layer=0, head=0)
    assert 0 <= result["diagonal_score"] <= 1
    assert isinstance(result["is_diagonal"], bool)


def test_diagonal_motif_self(model, tokens):
    result = detect_diagonal_motif(model, tokens, layer=0, head=0)
    assert 0 <= result["self_attention_score"] <= 1


def test_stripe_motif_structure(model, tokens):
    result = detect_stripe_motif(model, tokens, layer=0, head=0)
    assert 0 <= result["stripe_score"] <= 1
    assert isinstance(result["is_stripe"], bool)


def test_stripe_motif_column(model, tokens):
    result = detect_stripe_motif(model, tokens, layer=0, head=0)
    assert 0 <= result["dominant_column"] < 5
    assert 0 <= result["column_concentration"] <= 1


def test_block_motif_structure(model, tokens):
    result = detect_block_motif(model, tokens, layer=0, head=0)
    assert 0 <= result["local_score"] <= 1
    assert isinstance(result["is_local"], bool)


def test_triangular_motif_structure(model, tokens):
    result = detect_triangular_motif(model, tokens, layer=0, head=0)
    assert -1 <= result["triangular_score"] <= 1
    assert isinstance(result["is_triangular"], bool)


def test_motif_summary_structure(model, tokens):
    result = attention_motif_summary(model, tokens)
    assert len(result["classifications"]) == 8  # 2 layers * 4 heads


def test_motif_summary_categories(model, tokens):
    result = attention_motif_summary(model, tokens)
    for c in result["classifications"]:
        assert c["dominant_motif"] in ("diagonal", "stripe", "local", "triangular")


def test_motif_summary_counts(model, tokens):
    result = attention_motif_summary(model, tokens)
    assert sum(result["motif_counts"].values()) == 8
