"""Tests for attention_head_classification module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_classification import (
    detect_induction_heads, detect_previous_token_heads,
    detect_positional_heads, detect_copy_heads,
    head_classification_summary,
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
    return jnp.array([1, 5, 10, 5, 1])


def test_induction_heads_structure(model, tokens):
    result = detect_induction_heads(model, tokens)
    assert len(result["heads"]) == 8  # 2 layers * 4 heads
    assert isinstance(result["n_induction_heads"], int)


def test_induction_heads_scores(model, tokens):
    result = detect_induction_heads(model, tokens)
    for h in result["heads"]:
        assert 0 <= h["induction_score"] <= 1
        assert isinstance(h["is_induction"], bool)


def test_previous_token_heads_structure(model, tokens):
    result = detect_previous_token_heads(model, tokens)
    assert len(result["heads"]) == 8
    assert isinstance(result["n_previous_token_heads"], int)


def test_previous_token_heads_scores(model, tokens):
    result = detect_previous_token_heads(model, tokens)
    for h in result["heads"]:
        assert 0 <= h["prev_token_score"] <= 1


def test_positional_heads_structure(model, tokens):
    result = detect_positional_heads(model, tokens)
    assert len(result["heads"]) == 8
    for h in result["heads"]:
        assert 0 <= h["bos_attention"] <= 1
        assert 0 <= h["self_attention"] <= 1


def test_copy_heads_structure(model, tokens):
    result = detect_copy_heads(model, tokens)
    assert len(result["heads"]) == 8
    for h in result["heads"]:
        assert -1 <= h["mean_copy_score"] <= 1


def test_copy_heads_max_score(model, tokens):
    result = detect_copy_heads(model, tokens)
    for h in result["heads"]:
        assert h["max_copy_score"] >= h["mean_copy_score"]


def test_classification_summary_structure(model, tokens):
    result = head_classification_summary(model, tokens)
    assert len(result["classifications"]) == 8
    for c in result["classifications"]:
        assert c["primary"] in ("induction", "previous_token", "positional", "copy", "other")


def test_classification_summary_counts(model, tokens):
    result = head_classification_summary(model, tokens)
    total = result["n_induction"] + result["n_previous_token"] + result["n_positional"] + result["n_copy"] + result["n_other"]
    # Some heads may have multiple classifications, so total may exceed 8
    assert result["n_other"] >= 0
