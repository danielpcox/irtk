"""Tests for compositional structure discovery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.compositional_structure import (
    information_bottleneck_layers,
    subroutine_clustering,
    skip_connection_importance,
    algorithmic_decomposition,
    generalization_phase_analysis,
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


# ─── Information Bottleneck Layers ─────────────────────────────────────────


class TestInformationBottleneckLayers:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = information_bottleneck_layers(model, seqs)
        assert "effective_dims" in result
        assert "bottleneck_layers" in result
        assert "compression_ratios" in result

    def test_dims_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = information_bottleneck_layers(model, seqs)
        assert len(result["effective_dims"]) == 3  # n_layers+1

    def test_compression_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = information_bottleneck_layers(model, seqs)
        assert len(result["compression_ratios"]) == 2  # n_layers

    def test_dims_positive(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = information_bottleneck_layers(model, seqs)
        assert np.all(result["effective_dims"] >= 0)

    def test_tightest_valid(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = information_bottleneck_layers(model, seqs)
        assert 0 <= result["tightest_bottleneck"] < 3


# ─── Subroutine Clustering ────────────────────────────────────────────────


class TestSubroutineClustering:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subroutine_clustering(model, seqs, n_clusters=2)
        assert "cluster_assignments" in result
        assert "head_labels" in result
        assert "similarity_matrix" in result

    def test_assignments_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subroutine_clustering(model, seqs, n_clusters=2)
        # 2 layers * 4 heads = 8 total heads
        assert len(result["cluster_assignments"]) == 8

    def test_labels_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subroutine_clustering(model, seqs, n_clusters=2)
        assert len(result["head_labels"]) == 8

    def test_similarity_matrix_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subroutine_clustering(model, seqs, n_clusters=2)
        assert result["similarity_matrix"].shape == (8, 8)

    def test_cluster_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subroutine_clustering(model, seqs, n_clusters=3)
        assert np.all(result["cluster_assignments"] >= 0)
        assert np.all(result["cluster_assignments"] < 3)


# ─── Skip Connection Importance ────────────────────────────────────────────


class TestSkipConnectionImportance:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = skip_connection_importance(model, seqs, _metric)
        assert "skip_importance" in result
        assert "layer_importance" in result
        assert "skip_fraction" in result

    def test_importance_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = skip_connection_importance(model, seqs, _metric)
        assert len(result["skip_importance"]) == 2
        assert len(result["layer_importance"]) == 2

    def test_fraction_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = skip_connection_importance(model, seqs, _metric)
        for f in result["skip_fraction"]:
            assert 0 <= f <= 1.0

    def test_most_skip_dependent_valid(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = skip_connection_importance(model, seqs, _metric)
        assert 0 <= result["most_skip_dependent"] < 2


# ─── Algorithmic Decomposition ─────────────────────────────────────────────


class TestAlgorithmicDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = algorithmic_decomposition(model, tokens, _metric)
        assert "cumulative_metric" in result
        assert "phase_boundaries" in result
        assert "n_phases" in result
        assert "phase_contributions" in result

    def test_cumulative_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = algorithmic_decomposition(model, tokens, _metric)
        assert len(result["cumulative_metric"]) == 3  # n_layers+1

    def test_n_phases_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = algorithmic_decomposition(model, tokens, _metric)
        assert result["n_phases"] >= 1

    def test_last_cumulative_equals_full(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = algorithmic_decomposition(model, tokens, _metric)
        full_value = _metric(model(tokens))
        assert abs(result["cumulative_metric"][-1] - full_value) < 1e-4


# ─── Generalization Phase Analysis ─────────────────────────────────────────


class TestGeneralizationPhaseAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = generalization_phase_analysis(model, seqs, _metric)
        assert "group_effects" in result
        assert "group_std" in result
        assert "group_labels" in result

    def test_custom_groups(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        groups = [(0, 1), (1, 2)]
        result = generalization_phase_analysis(model, seqs, _metric, layer_groups=groups)
        assert len(result["group_effects"]) == 2
        assert len(result["group_labels"]) == 2

    def test_most_important_valid(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        groups = [(0, 1), (1, 2)]
        result = generalization_phase_analysis(model, seqs, _metric, layer_groups=groups)
        assert 0 <= result["most_important_group"] < 2

    def test_multiple_prompts(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = generalization_phase_analysis(model, seqs, _metric)
        # With multiple prompts, should have non-zero std
        assert len(result["group_std"]) > 0
