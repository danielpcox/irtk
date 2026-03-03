"""Tests for model_internals_dashboard module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_internals_dashboard import (
    layer_statistics,
    head_classification_summary,
    mlp_utilization,
    residual_stream_health,
    bottleneck_detection,
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


class TestLayerStatistics:
    def test_basic(self, model, tokens):
        result = layer_statistics(model, tokens)
        assert "per_layer" in result
        assert "summary" in result

    def test_per_layer_count(self, model, tokens):
        result = layer_statistics(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers

    def test_summary_keys(self, model, tokens):
        result = layer_statistics(model, tokens)
        assert "max_resid_layer" in result["summary"]
        assert "resid_growth" in result["summary"]


class TestHeadClassificationSummary:
    def test_basic(self, model, tokens):
        result = head_classification_summary(model, tokens)
        assert "classifications" in result
        assert "category_counts" in result
        assert "top_heads_per_category" in result

    def test_all_heads_classified(self, model, tokens):
        result = head_classification_summary(model, tokens)
        n_total = model.cfg.n_layers * model.cfg.n_heads
        assert len(result["classifications"]) == n_total

    def test_category_counts_sum(self, model, tokens):
        result = head_classification_summary(model, tokens)
        total = sum(result["category_counts"].values())
        assert total == model.cfg.n_layers * model.cfg.n_heads


class TestMlpUtilization:
    def test_basic(self, model, tokens):
        result = mlp_utilization(model, tokens)
        assert "per_layer" in result
        assert "overall_dead_fraction" in result
        assert "overall_sparsity" in result

    def test_per_layer_count(self, model, tokens):
        result = mlp_utilization(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers

    def test_fractions_range(self, model, tokens):
        result = mlp_utilization(model, tokens)
        assert 0 <= result["overall_dead_fraction"] <= 1
        assert 0 <= result["overall_sparsity"] <= 1


class TestResidualStreamHealth:
    def test_basic(self, model, tokens):
        result = residual_stream_health(model, tokens)
        assert "norm_trajectory" in result
        assert "rank_trajectory" in result
        assert "component_balance" in result
        assert "health_warnings" in result

    def test_trajectory_length(self, model, tokens):
        result = residual_stream_health(model, tokens)
        assert len(result["norm_trajectory"]) == model.cfg.n_layers

    def test_norms_positive(self, model, tokens):
        result = residual_stream_health(model, tokens)
        assert all(float(n) > 0 for n in result["norm_trajectory"])


class TestBottleneckDetection:
    def test_basic(self, model, tokens):
        result = bottleneck_detection(model, tokens)
        assert "bottleneck_layers" in result
        assert "rank_profile" in result
        assert "min_rank_layer" in result

    def test_rank_profile_length(self, model, tokens):
        result = bottleneck_detection(model, tokens)
        assert len(result["rank_profile"]) == model.cfg.n_layers

    def test_with_metric(self, model, tokens):
        result = bottleneck_detection(model, tokens, metric_fn=metric_fn)
        assert len(result["ablation_profile"]) == model.cfg.n_layers
