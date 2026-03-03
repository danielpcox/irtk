"""Tests for logit_attribution_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_attribution_analysis import (
    full_logit_decomposition, per_head_logit_attribution,
    logit_attribution_by_position, top_promoted_suppressed,
    logit_attribution_summary,
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


def test_full_decomposition_structure(model, tokens):
    result = full_logit_decomposition(model, tokens)
    # embed + 2*(attn+mlp) + bias = 1 + 4 + 1 = 6
    assert len(result["components"]) == 6


def test_full_decomposition_names(model, tokens):
    result = full_logit_decomposition(model, tokens)
    names = [c["name"] for c in result["components"]]
    assert "embed" in names
    assert "bias" in names
    assert "L0_attn" in names


def test_per_head_attribution_structure(model, tokens):
    result = per_head_logit_attribution(model, tokens)
    assert len(result["heads"]) == 8  # 2 layers * 4 heads


def test_per_head_attribution_sorted(model, tokens):
    result = per_head_logit_attribution(model, tokens)
    abs_contribs = [abs(h["logit_contribution"]) for h in result["heads"]]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_attribution_by_position_structure(model, tokens):
    result = logit_attribution_by_position(model, tokens)
    assert len(result["per_source"]) == 5


def test_attribution_by_position_sorted(model, tokens):
    result = logit_attribution_by_position(model, tokens)
    abs_contribs = [abs(s["logit_contribution"]) for s in result["per_source"]]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_top_promoted_suppressed_structure(model, tokens):
    result = top_promoted_suppressed(model, tokens, top_k=5)
    assert len(result["promoted"]) == 5
    assert len(result["suppressed"]) == 5
    assert result["logit_range"] >= 0


def test_top_promoted_sorted(model, tokens):
    result = top_promoted_suppressed(model, tokens, top_k=5)
    logits = [t["logit"] for t in result["promoted"]]
    assert logits == sorted(logits, reverse=True)


def test_attribution_summary_structure(model, tokens):
    result = logit_attribution_summary(model, tokens)
    assert isinstance(result["total_logit"], float)
    assert isinstance(result["total_attn_contribution"], float)
    assert isinstance(result["total_mlp_contribution"], float)
