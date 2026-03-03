"""Tests for model_comparison module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_comparison import (
    weight_distance,
    activation_divergence,
    prediction_agreement,
    attention_pattern_comparison,
    component_importance_comparison,
)


@pytest.fixture
def model_a():
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
def model_b():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(99)
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


class TestWeightDistance:
    def test_basic(self, model_a, model_b):
        result = weight_distance(model_a, model_b)
        assert "layer_distances" in result
        assert "total_distance" in result
        assert "max_distance_component" in result

    def test_self_zero(self, model_a):
        result = weight_distance(model_a, model_a)
        assert result["total_distance"] < 1e-6

    def test_distances_nonneg(self, model_a, model_b):
        result = weight_distance(model_a, model_b)
        for d in result["layer_distances"].values():
            assert d >= 0


class TestActivationDivergence:
    def test_basic(self, model_a, model_b, tokens):
        result = activation_divergence(model_a, model_b, tokens)
        assert "layer_divergence" in result
        assert "cosine_similarity" in result
        assert "logit_divergence" in result

    def test_shapes(self, model_a, model_b, tokens):
        result = activation_divergence(model_a, model_b, tokens)
        nl = model_a.cfg.n_layers
        assert result["layer_divergence"].shape == (nl,)
        assert result["cosine_similarity"].shape == (nl,)

    def test_self_zero(self, model_a, tokens):
        result = activation_divergence(model_a, model_a, tokens)
        assert result["logit_divergence"] < 1e-6


class TestPredictionAgreement:
    def test_basic(self, model_a, model_b, tokens):
        result = prediction_agreement(model_a, model_b, [tokens])
        assert "agreement_rate" in result
        assert "top_k_overlap" in result
        assert "kl_divergences" in result
        assert "mean_kl" in result

    def test_self_agreement(self, model_a, tokens):
        result = prediction_agreement(model_a, model_a, [tokens])
        assert result["agreement_rate"] == 1.0

    def test_ranges(self, model_a, model_b, tokens):
        result = prediction_agreement(model_a, model_b, [tokens])
        assert 0 <= result["agreement_rate"] <= 1
        assert 0 <= result["top_k_overlap"] <= 1


class TestAttentionPatternComparison:
    def test_basic(self, model_a, model_b, tokens):
        result = attention_pattern_comparison(model_a, model_b, tokens)
        assert "pattern_distances" in result
        assert "pattern_cosine" in result
        assert "most_different_head" in result

    def test_shapes(self, model_a, model_b, tokens):
        result = attention_pattern_comparison(model_a, model_b, tokens)
        nl, nh = model_a.cfg.n_layers, model_a.cfg.n_heads
        assert result["pattern_distances"].shape == (nl, nh)
        assert result["pattern_cosine"].shape == (nl, nh)

    def test_self_zero(self, model_a, tokens):
        result = attention_pattern_comparison(model_a, model_a, tokens)
        assert result["mean_distance"] < 1e-6


class TestComponentImportanceComparison:
    def test_basic(self, model_a, model_b, tokens, metric_fn):
        result = component_importance_comparison(model_a, model_b, tokens, metric_fn)
        assert "importance_a" in result
        assert "importance_b" in result
        assert "rank_correlation" in result
        assert "biggest_rank_changes" in result

    def test_self_perfect_correlation(self, model_a, tokens, metric_fn):
        result = component_importance_comparison(model_a, model_a, tokens, metric_fn)
        assert result["rank_correlation"] > 0.99

    def test_correlation_range(self, model_a, model_b, tokens, metric_fn):
        result = component_importance_comparison(model_a, model_b, tokens, metric_fn)
        assert -1.01 <= result["rank_correlation"] <= 1.01
