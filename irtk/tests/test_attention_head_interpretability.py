"""Tests for attention_head_interpretability module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_interpretability import (
    head_function_classification,
    entropy_behavior_mapping,
    qk_ov_summary,
    importance_interpretability_tradeoff,
    head_summary_card,
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
    return jnp.array([0, 5, 10, 15, 20])


@pytest.fixture
def metric_fn():
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


class TestHeadFunctionClassification:
    def test_basic(self, model, tokens):
        result = head_function_classification(model, tokens)
        assert "classifications" in result
        assert "scores" in result
        assert "function_counts" in result
        assert "confidence" in result

    def test_all_heads_classified(self, model, tokens):
        result = head_function_classification(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert len(result["classifications"]) == nl * nh

    def test_valid_labels(self, model, tokens):
        result = head_function_classification(model, tokens)
        valid_labels = {"previous_token", "self_attention", "bos_attention",
                       "induction", "local_window", "distributed"}
        for label in result["classifications"].values():
            assert label in valid_labels

    def test_counts_sum(self, model, tokens):
        result = head_function_classification(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert sum(result["function_counts"].values()) == nl * nh


class TestEntropyBehaviorMapping:
    def test_basic(self, model, tokens):
        result = entropy_behavior_mapping(model, tokens)
        assert "head_entropy" in result
        assert "entropy_categories" in result
        assert "focus_positions" in result
        assert "entropy_variance" in result

    def test_shapes(self, model, tokens):
        result = entropy_behavior_mapping(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["head_entropy"].shape == (nl, nh)
        assert result["entropy_variance"].shape == (nl, nh)

    def test_entropy_nonneg(self, model, tokens):
        result = entropy_behavior_mapping(model, tokens)
        assert np.all(result["head_entropy"] >= 0)

    def test_valid_categories(self, model, tokens):
        result = entropy_behavior_mapping(model, tokens)
        for cat in result["entropy_categories"].values():
            assert cat in {"focused", "moderate", "diffuse"}


class TestQkOvSummary:
    def test_basic(self, model):
        result = qk_ov_summary(model, layer=0, head=0)
        assert "qk_top_interactions" in result
        assert "ov_top_mappings" in result
        assert "qk_rank" in result
        assert "ov_rank" in result

    def test_ranks_positive(self, model):
        result = qk_ov_summary(model, layer=0, head=0)
        assert result["qk_rank"] >= 0
        assert result["ov_rank"] >= 0

    def test_singular_values(self, model):
        result = qk_ov_summary(model, layer=0, head=0)
        assert len(result["qk_singular_values"]) > 0
        assert len(result["ov_singular_values"]) > 0


class TestImportanceInterpretabilityTradeoff:
    def test_basic(self, model, tokens, metric_fn):
        result = importance_interpretability_tradeoff(model, tokens, metric_fn)
        assert "importance" in result
        assert "interpretability" in result
        assert "tradeoff_correlation" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = importance_interpretability_tradeoff(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["importance"].shape == (nl, nh)
        assert result["interpretability"].shape == (nl, nh)

    def test_correlation_range(self, model, tokens, metric_fn):
        result = importance_interpretability_tradeoff(model, tokens, metric_fn)
        assert -1.01 <= result["tradeoff_correlation"] <= 1.01


class TestHeadSummaryCard:
    def test_basic(self, model, tokens):
        result = head_summary_card(model, tokens, layer=0, head=0)
        assert "identity" in result
        assert "function_type" in result
        assert "entropy_category" in result
        assert "qk_rank" in result
        assert "ov_rank" in result
        assert "clarity" in result

    def test_with_metric(self, model, tokens, metric_fn):
        result = head_summary_card(model, tokens, layer=0, head=0, metric_fn=metric_fn)
        assert result["importance"] >= 0

    def test_identity(self, model, tokens):
        result = head_summary_card(model, tokens, layer=1, head=2)
        assert result["identity"] == (1, 2)
