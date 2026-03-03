"""Tests for feature deletion sensitivity analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.feature_deletion_sensitivity import (
    feature_importance_ranking,
    feature_deletion_cascade,
    feature_interaction_effects,
    minimal_sufficient_features,
    feature_redundancy_clusters,
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


def _metric(logits):
    return float(logits[-1, 0])


def _make_features(n=5, d=16, seed=123):
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / (norms + 1e-10)


class TestFeatureImportanceRanking:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_importance_ranking(model, tokens, features, _metric)
        assert "importance_scores" in result
        assert "ranking" in result
        assert "baseline_metric" in result
        assert "top_feature" in result

    def test_scores_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_importance_ranking(model, tokens, features, _metric)
        assert np.all(result["importance_scores"] >= 0)

    def test_ranking_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(4, 16)
        result = feature_importance_ranking(model, tokens, features, _metric)
        assert len(result["ranking"]) == 4


class TestFeatureDeletionCascade:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_deletion_cascade(model, tokens, features, _metric)
        assert "deletion_order" in result
        assert "cumulative_metrics" in result
        assert "metric_at_half" in result
        assert "features_for_90pct" in result

    def test_metrics_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_deletion_cascade(model, tokens, features, _metric)
        assert len(result["cumulative_metrics"]) == 4  # n_features + 1


class TestFeatureInteractionEffects:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_interaction_effects(model, tokens, features, _metric)
        assert "interaction_matrix" in result
        assert "pair_indices" in result
        assert "strongest_interaction" in result
        assert "mean_interaction" in result

    def test_mean_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = feature_interaction_effects(model, tokens, features, _metric)
        assert result["mean_interaction"] >= 0


class TestMinimalSufficientFeatures:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = minimal_sufficient_features(model, tokens, features, _metric)
        assert "sufficient_features" in result
        assert "n_sufficient" in result
        assert "fraction_of_total" in result
        assert "achieved_metric" in result

    def test_sufficient_subset(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        features = _make_features(3, 16)
        result = minimal_sufficient_features(model, tokens, features, _metric)
        assert result["n_sufficient"] <= 3


class TestFeatureRedundancyClusters:
    def test_returns_dict(self):
        model = _make_model()
        features = _make_features(6, 16)
        result = feature_redundancy_clusters(model, features, n_clusters=2)
        assert "cluster_assignments" in result
        assert "cluster_sizes" in result
        assert "within_cluster_similarity" in result
        assert "between_cluster_similarity" in result

    def test_assignments_length(self):
        model = _make_model()
        features = _make_features(6, 16)
        result = feature_redundancy_clusters(model, features, n_clusters=2)
        assert len(result["cluster_assignments"]) == 6

    def test_sizes_sum(self):
        model = _make_model()
        features = _make_features(6, 16)
        result = feature_redundancy_clusters(model, features, n_clusters=2)
        assert np.sum(result["cluster_sizes"]) == 6
