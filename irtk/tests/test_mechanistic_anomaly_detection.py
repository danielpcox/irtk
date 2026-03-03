"""Tests for mechanistic anomaly detection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.mechanistic_anomaly_detection import (
    build_activation_profile,
    detect_pathway_anomalies,
    find_trojan_signatures,
    compare_surface_vs_internals,
    cluster_computational_strategies,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestBuildActivationProfile:
    def test_returns_dict(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = build_activation_profile(model, refs)
        assert "means" in result
        assert "stds" in result
        assert "covariances" in result
        assert result["n_samples"] == 3

    def test_profile_shapes(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = build_activation_profile(model, refs)
        for name in result["hook_names"]:
            assert result["means"][name].shape == (16,)
            assert result["stds"][name].shape == (16,)
            assert result["covariances"][name].shape == (16, 16)

    def test_hook_names_default(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3])]
        result = build_activation_profile(model, refs)
        assert len(result["hook_names"]) == 2  # n_layers


class TestDetectPathwayAnomalies:
    def test_returns_dict(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        profile = build_activation_profile(model, refs)
        tokens = jnp.array([10, 20, 30, 40])
        result = detect_pathway_anomalies(model, tokens, profile)
        assert "layer_anomaly_scores" in result
        assert "max_anomaly_layer" in result
        assert "total_anomaly_score" in result

    def test_anomaly_score_nonneg(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        profile = build_activation_profile(model, refs)
        tokens = jnp.array([10, 20, 30, 40])
        result = detect_pathway_anomalies(model, tokens, profile)
        assert result["total_anomaly_score"] >= 0


class TestFindTrojanSignatures:
    def test_returns_dict(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        profile = build_activation_profile(model, refs)
        tokens = jnp.array([10, 20, 30, 40])
        result = find_trojan_signatures(model, tokens, profile)
        assert "suspicious_dimensions" in result
        assert "sparsity_ratios" in result
        assert "trojan_risk_score" in result

    def test_risk_score_nonneg(self):
        model = _make_model()
        refs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        profile = build_activation_profile(model, refs)
        tokens = jnp.array([10, 20, 30, 40])
        result = find_trojan_signatures(model, tokens, profile)
        assert result["trojan_risk_score"] >= 0


class TestCompareSurfaceVsInternals:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compare_surface_vs_internals(model, tokens)
        assert "layer_predictions" in result
        assert "final_prediction" in result
        assert "agreement_fraction" in result
        assert "trajectory_consistency" in result

    def test_predictions_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compare_surface_vs_internals(model, tokens)
        assert len(result["layer_predictions"]) == 2  # n_layers

    def test_agreement_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compare_surface_vs_internals(model, tokens)
        assert 0.0 <= result["agreement_fraction"] <= 1.0


class TestClusterComputationalStrategies:
    def test_returns_dict(self):
        model = _make_model()
        inputs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                  jnp.array([8, 9, 10, 11]), jnp.array([12, 13, 14, 15])]
        result = cluster_computational_strategies(model, inputs, n_clusters=2)
        assert "cluster_assignments" in result
        assert "cluster_sizes" in result
        assert "cluster_centroids" in result
        assert "between_cluster_variance" in result

    def test_assignments_length(self):
        model = _make_model()
        inputs = [jnp.array([i, i+1, i+2, i+3]) for i in range(6)]
        result = cluster_computational_strategies(model, inputs, n_clusters=2)
        assert len(result["cluster_assignments"]) == 6

    def test_cluster_sizes_sum(self):
        model = _make_model()
        inputs = [jnp.array([i, i+1, i+2, i+3]) for i in range(6)]
        result = cluster_computational_strategies(model, inputs, n_clusters=2)
        assert np.sum(result["cluster_sizes"]) == 6
