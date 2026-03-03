"""Tests for feature_circuits module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.feature_circuits import (
    feature_propagation_trace,
    feature_composition_scores,
    feature_path_attribution,
    feature_interaction_matrix,
    feature_logit_effect,
)


def _make_model(seed=42):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(seed)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


@pytest.fixture
def direction():
    rng = np.random.RandomState(42)
    d = rng.randn(16).astype(np.float32)
    return d / np.linalg.norm(d)


class TestFeaturePropagationTrace:
    def test_output_keys(self, model, tokens, direction):
        r = feature_propagation_trace(model, tokens, direction)
        assert "projections" in r
        assert "attn_contributions" in r
        assert "mlp_contributions" in r
        assert "peak_layer" in r
        assert "emergence_layer" in r

    def test_shapes(self, model, tokens, direction):
        r = feature_propagation_trace(model, tokens, direction)
        n_layers = model.cfg.n_layers
        assert r["projections"].shape == (n_layers + 1,)
        assert r["attn_contributions"].shape == (n_layers,)
        assert r["mlp_contributions"].shape == (n_layers,)

    def test_peak_valid(self, model, tokens, direction):
        r = feature_propagation_trace(model, tokens, direction)
        assert 0 <= r["peak_layer"] <= model.cfg.n_layers


class TestFeatureCompositionScores:
    def test_output_keys(self, model, direction):
        d2 = np.random.randn(16).astype(np.float32)
        r = feature_composition_scores(model, direction, d2)
        assert "ov_scores" in r
        assert "qk_scores" in r
        assert "max_ov_head" in r
        assert "max_qk_head" in r
        assert "total_ov_score" in r

    def test_shapes(self, model, direction):
        d2 = np.random.randn(16).astype(np.float32)
        r = feature_composition_scores(model, direction, d2)
        assert r["ov_scores"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["qk_scores"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_total_nonneg(self, model, direction):
        d2 = np.random.randn(16).astype(np.float32)
        r = feature_composition_scores(model, direction, d2)
        assert r["total_ov_score"] >= 0


class TestFeaturePathAttribution:
    def test_output_keys(self, model, tokens, direction):
        def metric(logits): return float(logits[-1, 0])
        r = feature_path_attribution(model, tokens, direction, metric)
        assert "baseline_metric" in r
        assert "attn_attributions" in r
        assert "mlp_attributions" in r
        assert "dominant_path" in r

    def test_shapes(self, model, tokens, direction):
        def metric(logits): return float(logits[-1, 0])
        r = feature_path_attribution(model, tokens, direction, metric)
        assert r["attn_attributions"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["mlp_attributions"].shape == (model.cfg.n_layers,)

    def test_attributions_nonneg(self, model, tokens, direction):
        def metric(logits): return float(logits[-1, 0])
        r = feature_path_attribution(model, tokens, direction, metric)
        assert np.all(r["attn_attributions"] >= 0)
        assert np.all(r["mlp_attributions"] >= 0)


class TestFeatureInteractionMatrix:
    def test_output_keys(self, model, tokens):
        rng = np.random.RandomState(42)
        dirs = [rng.randn(16).astype(np.float32) for _ in range(3)]
        r = feature_interaction_matrix(model, dirs, tokens)
        assert "coactivation_matrix" in r
        assert "ov_interaction_matrix" in r
        assert "most_interacting_pair" in r
        assert "mean_interaction" in r

    def test_shapes(self, model, tokens):
        rng = np.random.RandomState(42)
        dirs = [rng.randn(16).astype(np.float32) for _ in range(3)]
        r = feature_interaction_matrix(model, dirs, tokens)
        assert r["coactivation_matrix"].shape == (3, 3)
        assert r["ov_interaction_matrix"].shape == (3, 3)

    def test_interaction_nonneg(self, model, tokens):
        rng = np.random.RandomState(42)
        dirs = [rng.randn(16).astype(np.float32) for _ in range(3)]
        r = feature_interaction_matrix(model, dirs, tokens)
        assert r["mean_interaction"] >= 0


class TestFeatureLogitEffect:
    def test_output_keys(self, model, tokens, direction):
        r = feature_logit_effect(model, tokens, direction, top_k=3)
        assert "logit_effects" in r
        assert "top_promoted" in r
        assert "top_demoted" in r
        assert "promotion_scores" in r
        assert "demotion_scores" in r

    def test_shapes(self, model, tokens, direction):
        r = feature_logit_effect(model, tokens, direction, top_k=3)
        assert r["logit_effects"].shape == (model.cfg.d_vocab,)
        assert len(r["top_promoted"]) == 3
        assert len(r["top_demoted"]) == 3

    def test_promoted_greater_than_demoted(self, model, tokens, direction):
        r = feature_logit_effect(model, tokens, direction, top_k=3)
        assert r["promotion_scores"][0] >= r["demotion_scores"][0]
