"""Tests for attention_gradient_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_gradient_analysis import (
    attention_gradient_attribution,
    gradient_weighted_pattern,
    attention_sensitivity_map,
    gradient_head_ranking,
    attention_gradient_flow,
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


def metric_fn(logits, tokens):
    return jnp.mean(logits[-1])


class TestAttentionGradientAttribution:
    def test_basic(self, model, tokens):
        result = attention_gradient_attribution(model, tokens, metric_fn, layer=0, head=0)
        assert "attention_gradient" in result
        assert "gradient_magnitude" in result
        assert "top_entries" in result

    def test_gradient_shape(self, model, tokens):
        result = attention_gradient_attribution(model, tokens, metric_fn, layer=0, head=0)
        seq_len = len(tokens)
        assert result["attention_gradient"].shape == (seq_len, seq_len)


class TestGradientWeightedPattern:
    def test_basic(self, model, tokens):
        result = gradient_weighted_pattern(model, tokens, metric_fn, layer=0)
        assert "weighted_pattern" in result
        assert "head_importances" in result
        assert "head_weights" in result

    def test_weights_sum(self, model, tokens):
        result = gradient_weighted_pattern(model, tokens, metric_fn, layer=0)
        assert abs(float(jnp.sum(result["head_weights"])) - 1.0) < 0.01


class TestAttentionSensitivityMap:
    def test_basic(self, model, tokens):
        result = attention_sensitivity_map(model, tokens, metric_fn, layer=0, head=0)
        assert "position_sensitivity" in result
        assert "most_sensitive_position" in result

    def test_shape(self, model, tokens):
        result = attention_sensitivity_map(model, tokens, metric_fn, layer=0, head=0)
        assert result["position_sensitivity"].shape == (len(tokens),)


class TestGradientHeadRanking:
    def test_basic(self, model, tokens):
        result = gradient_head_ranking(model, tokens, metric_fn)
        assert "head_ranking" in result
        assert "importance_matrix" in result
        assert "top_heads" in result

    def test_matrix_shape(self, model, tokens):
        result = gradient_head_ranking(model, tokens, metric_fn)
        assert result["importance_matrix"].shape == (model.cfg.n_layers, model.cfg.n_heads)


class TestAttentionGradientFlow:
    def test_basic(self, model, tokens):
        result = attention_gradient_flow(model, tokens, metric_fn)
        assert "attn_importance" in result
        assert "mlp_importance" in result
        assert "cumulative_importance" in result
        assert "peak_layer" in result

    def test_shapes(self, model, tokens):
        result = attention_gradient_flow(model, tokens, metric_fn)
        nl = model.cfg.n_layers
        assert result["attn_importance"].shape == (nl,)
        assert result["mlp_importance"].shape == (nl,)
