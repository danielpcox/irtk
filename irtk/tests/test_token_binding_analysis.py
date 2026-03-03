"""Tests for token_binding_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_binding_analysis import (
    token_role_binding, binding_strength_comparison,
    binding_source_attribution, binding_competition,
    token_binding_summary,
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


def test_role_binding_structure(model, tokens):
    result = token_role_binding(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_strongly_bound"], bool)


def test_role_binding_cosine(model, tokens):
    result = token_role_binding(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["cosine_to_embed"] <= 1
        assert p["divergence"] >= 0


def test_binding_comparison_structure(model, tokens):
    result = binding_strength_comparison(model, tokens)
    assert len(result["per_position"]) == 5


def test_binding_comparison_sorted(model, tokens):
    result = binding_strength_comparison(model, tokens)
    divs = [p["divergence"] for p in result["per_position"]]
    assert divs == sorted(divs, reverse=True)


def test_source_attribution_structure(model, tokens):
    result = binding_source_attribution(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["binding_source"] in ("attention", "mlp")


def test_source_attribution_norms(model, tokens):
    result = binding_source_attribution(model, tokens)
    for p in result["per_layer"]:
        assert p["attn_divergence"] >= 0
        assert p["mlp_divergence"] >= 0


def test_binding_competition_structure(model, tokens):
    result = binding_competition(model, tokens, position=-1, top_k=3)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert len(p["competitors"]) == 3


def test_binding_competition_cosine(model, tokens):
    result = binding_competition(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["top_cosine"] <= 1


def test_binding_summary_structure(model, tokens):
    result = token_binding_summary(model, tokens)
    assert len(result["per_position"]) == 5
    assert isinstance(result["mean_binding_rate"], float)
