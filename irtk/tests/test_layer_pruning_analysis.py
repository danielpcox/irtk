"""Tests for layer_pruning_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_pruning_analysis import (
    layer_skip_analysis,
    progressive_layer_pruning,
    layer_criticality_profile,
    layer_similarity_for_pruning,
    optimal_layer_subset,
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
    def fn(logits):
        return float(logits[-1, 0] - logits[-1, 1])
    return fn


class TestLayerSkipAnalysis:
    def test_basic(self, model, tokens, metric_fn):
        result = layer_skip_analysis(model, tokens, metric_fn)
        assert "skip_effects" in result
        assert "most_critical_layer" in result
        assert "least_critical_layer" in result
        assert "mean_skip_effect" in result
        assert "can_skip" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = layer_skip_analysis(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        assert result["skip_effects"].shape == (n_layers,)
        assert result["can_skip"].shape == (n_layers,)
        assert 0 <= result["most_critical_layer"] < n_layers
        assert 0 <= result["least_critical_layer"] < n_layers

    def test_effects_nonnegative(self, model, tokens, metric_fn):
        result = layer_skip_analysis(model, tokens, metric_fn)
        assert np.all(result["skip_effects"] >= 0)


class TestProgressiveLayerPruning:
    def test_basic(self, model, tokens, metric_fn):
        result = progressive_layer_pruning(model, tokens, metric_fn)
        assert "pruning_order" in result
        assert "metrics_after_pruning" in result
        assert "layers_before_50pct_loss" in result
        assert "graceful_degradation" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = progressive_layer_pruning(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        assert len(result["pruning_order"]) == n_layers
        assert len(result["metrics_after_pruning"]) == n_layers + 1
        assert set(result["pruning_order"]) == set(range(n_layers))

    def test_layers_before_loss_valid(self, model, tokens, metric_fn):
        result = progressive_layer_pruning(model, tokens, metric_fn)
        assert 0 <= result["layers_before_50pct_loss"] <= model.cfg.n_layers


class TestLayerCriticalityProfile:
    def test_basic(self, model, tokens, metric_fn):
        result = layer_criticality_profile(model, tokens, metric_fn)
        assert "attn_criticality" in result
        assert "mlp_criticality" in result
        assert "combined_criticality" in result
        assert "critical_layers" in result
        assert "redundant_layers" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = layer_criticality_profile(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        assert result["attn_criticality"].shape == (n_layers,)
        assert result["mlp_criticality"].shape == (n_layers,)
        assert result["combined_criticality"].shape == (n_layers,)

    def test_combined_is_sum(self, model, tokens, metric_fn):
        result = layer_criticality_profile(model, tokens, metric_fn)
        np.testing.assert_allclose(
            result["combined_criticality"],
            result["attn_criticality"] + result["mlp_criticality"],
            atol=1e-6,
        )

    def test_criticality_nonnegative(self, model, tokens, metric_fn):
        result = layer_criticality_profile(model, tokens, metric_fn)
        assert np.all(result["attn_criticality"] >= 0)
        assert np.all(result["mlp_criticality"] >= 0)


class TestLayerSimilarityForPruning:
    def test_basic(self, model, tokens):
        result = layer_similarity_for_pruning(model, tokens)
        assert "layer_similarity_matrix" in result
        assert "most_similar_pair" in result
        assert "most_different_pair" in result
        assert "similarity_to_identity" in result

    def test_shapes(self, model, tokens):
        result = layer_similarity_for_pruning(model, tokens)
        n_layers = model.cfg.n_layers
        assert result["layer_similarity_matrix"].shape == (n_layers, n_layers)
        assert result["similarity_to_identity"].shape == (n_layers,)

    def test_self_similarity(self, model, tokens):
        result = layer_similarity_for_pruning(model, tokens)
        diag = np.diag(result["layer_similarity_matrix"])
        np.testing.assert_allclose(diag, 1.0, atol=0.01)


class TestOptimalLayerSubset:
    def test_basic(self, model, tokens, metric_fn):
        result = optimal_layer_subset(model, tokens, metric_fn)
        assert "kept_layers" in result
        assert "pruned_layers" in result
        assert "subset_metric" in result
        assert "full_metric" in result
        assert "retention_ratio" in result

    def test_default_target(self, model, tokens, metric_fn):
        result = optimal_layer_subset(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        expected_kept = max(1, n_layers // 2)
        assert len(result["kept_layers"]) == expected_kept
        assert len(result["pruned_layers"]) == n_layers - expected_kept
        assert set(result["kept_layers"] + result["pruned_layers"]) == set(range(n_layers))

    def test_custom_target(self, model, tokens, metric_fn):
        result = optimal_layer_subset(model, tokens, metric_fn, target_layers=1)
        assert len(result["kept_layers"]) == 1
        assert len(result["pruned_layers"]) == model.cfg.n_layers - 1

    def test_retention_ratio(self, model, tokens, metric_fn):
        result = optimal_layer_subset(model, tokens, metric_fn)
        assert result["retention_ratio"] >= 0
