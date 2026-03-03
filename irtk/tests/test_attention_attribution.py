"""Tests for attention_attribution module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_attribution import (
    attention_knockout_attribution,
    attention_value_decomposition,
    position_specific_attribution,
    attention_logit_contribution,
    attention_pattern_metric_sensitivity,
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


class TestAttentionKnockoutAttribution:
    def test_basic(self, model, tokens, metric_fn):
        result = attention_knockout_attribution(model, tokens, metric_fn)
        assert "head_effects" in result
        assert "total_attribution" in result
        assert "top_heads" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = attention_knockout_attribution(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["head_effects"].shape == (nl, nh)


class TestAttentionValueDecomposition:
    def test_basic(self, model, tokens):
        result = attention_value_decomposition(model, tokens, layer=0)
        assert "head_source_contributions" in result
        assert "top_sources_per_head" in result
        assert "head_output_norms" in result

    def test_shapes(self, model, tokens):
        result = attention_value_decomposition(model, tokens, layer=0)
        nh = model.cfg.n_heads
        seq_len = len(tokens)
        assert result["head_source_contributions"].shape == (nh, seq_len)
        assert result["head_output_norms"].shape == (nh,)

    def test_nonneg(self, model, tokens):
        result = attention_value_decomposition(model, tokens, layer=0)
        assert np.all(result["head_source_contributions"] >= 0)


class TestPositionSpecificAttribution:
    def test_basic(self, model, tokens, metric_fn):
        result = position_specific_attribution(model, tokens, metric_fn)
        assert "position_effects" in result
        assert "most_important_positions" in result
        assert "position_summary" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = position_specific_attribution(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        seq_len = len(tokens)
        assert result["position_effects"].shape == (nl, nh, seq_len)
        assert result["position_summary"].shape == (seq_len,)


class TestAttentionLogitContribution:
    def test_basic(self, model, tokens):
        result = attention_logit_contribution(model, tokens)
        assert "head_logit_contributions" in result
        assert "head_top_tokens" in result
        assert "total_attn_logit" in result

    def test_total_shape(self, model, tokens):
        result = attention_logit_contribution(model, tokens)
        assert result["total_attn_logit"].shape == (model.cfg.d_vocab,)

    def test_all_heads_present(self, model, tokens):
        result = attention_logit_contribution(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert len(result["head_logit_contributions"]) == nl * nh


class TestAttentionPatternMetricSensitivity:
    def test_basic(self, model, tokens, metric_fn):
        result = attention_pattern_metric_sensitivity(model, tokens, metric_fn, 0, 0)
        assert "base_metric" in result
        assert "noise_metrics" in result
        assert "sensitivity" in result
        assert "max_deviation" in result

    def test_sensitivity_nonneg(self, model, tokens, metric_fn):
        result = attention_pattern_metric_sensitivity(model, tokens, metric_fn, 0, 0)
        assert result["sensitivity"] >= 0
        assert result["max_deviation"] >= 0
