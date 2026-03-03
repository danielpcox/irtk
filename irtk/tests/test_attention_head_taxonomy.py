"""Tests for attention_head_taxonomy module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_taxonomy import (
    induction_head_score,
    previous_token_head_score,
    copy_head_score,
    inhibition_head_score,
    head_taxonomy_summary,
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
    # Include repeated tokens for induction scoring
    return jnp.array([0, 5, 10, 0, 5, 10])


@pytest.fixture
def metric_fn():
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


class TestInductionHeadScore:
    def test_basic(self, model, tokens):
        result = induction_head_score(model, tokens)
        assert "scores" in result
        assert "top_heads" in result
        assert "max_score" in result

    def test_shapes(self, model, tokens):
        result = induction_head_score(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["scores"].shape == (nl, nh)

    def test_nonneg(self, model, tokens):
        result = induction_head_score(model, tokens)
        assert np.all(result["scores"] >= 0)
        assert result["max_score"] >= 0


class TestPreviousTokenHeadScore:
    def test_basic(self, model, tokens):
        result = previous_token_head_score(model, tokens)
        assert "scores" in result
        assert "top_heads" in result
        assert "max_score" in result

    def test_shapes(self, model, tokens):
        result = previous_token_head_score(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["scores"].shape == (nl, nh)

    def test_range(self, model, tokens):
        result = previous_token_head_score(model, tokens)
        assert np.all(result["scores"] >= 0)
        assert np.all(result["scores"] <= 1.01)


class TestCopyHeadScore:
    def test_basic(self, model, tokens):
        result = copy_head_score(model, tokens)
        assert "scores" in result
        assert "top_heads" in result
        assert "copied_tokens" in result

    def test_shapes(self, model, tokens):
        result = copy_head_score(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["scores"].shape == (nl, nh)

    def test_range(self, model, tokens):
        result = copy_head_score(model, tokens)
        assert np.all(result["scores"] >= 0)
        assert np.all(result["scores"] <= 1.01)


class TestInhibitionHeadScore:
    def test_basic(self, model, tokens, metric_fn):
        result = inhibition_head_score(model, tokens, metric_fn)
        assert "scores" in result
        assert "top_heads" in result
        assert "n_inhibitory" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = inhibition_head_score(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["scores"].shape == (nl, nh)

    def test_nonneg(self, model, tokens, metric_fn):
        result = inhibition_head_score(model, tokens, metric_fn)
        assert np.all(result["scores"] >= 0)


class TestHeadTaxonomySummary:
    def test_basic(self, model, tokens, metric_fn):
        result = head_taxonomy_summary(model, tokens, metric_fn)
        assert "classifications" in result
        assert "type_counts" in result
        assert "type_distribution" in result
        assert "head_details" in result

    def test_all_heads_classified(self, model, tokens, metric_fn):
        result = head_taxonomy_summary(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert len(result["classifications"]) == nl * nh

    def test_distribution_sums(self, model, tokens, metric_fn):
        result = head_taxonomy_summary(model, tokens, metric_fn)
        assert abs(sum(result["type_distribution"].values()) - 1.0) < 0.01
