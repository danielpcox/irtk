"""Tests for attention_information_content module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_information_content import (
    attention_entropy_profile, attention_mutual_information,
    attention_concentration, information_flow_rate,
    information_content_summary,
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


def test_entropy_profile_structure(model, tokens):
    result = attention_entropy_profile(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["mean_entropy"] >= 0


def test_entropy_profile_values(model, tokens):
    result = attention_entropy_profile(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mean_entropy"] >= 0
        assert isinstance(h["is_sharp"], bool)


def test_mutual_information_structure(model, tokens):
    result = attention_mutual_information(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["mean_mi"] >= 0


def test_mutual_information_nonneg(model, tokens):
    result = attention_mutual_information(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["mutual_information"] >= 0


def test_concentration_structure(model, tokens):
    result = attention_concentration(model, tokens, layer=0, head=0)
    assert len(result["per_query"]) == 5
    assert isinstance(result["is_concentrated"], bool)


def test_concentration_mass(model, tokens):
    result = attention_concentration(model, tokens, layer=0, head=0)
    for pq in result["per_query"]:
        assert 0 <= pq["top1_mass"] <= 1
        assert pq["top3_mass"] >= pq["top1_mass"]


def test_flow_rate_structure(model, tokens):
    result = information_flow_rate(model, tokens, layer=0)
    assert len(result["per_head"]) == 4
    assert result["mean_flow"] >= 0


def test_summary_structure(model, tokens):
    result = information_content_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = information_content_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_entropy"] >= 0
        assert p["mean_mi"] >= 0
