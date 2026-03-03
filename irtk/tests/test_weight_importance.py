"""Tests for weight importance analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.weight_importance import (
    fisher_information_importance,
    activation_variance_importance,
    lottery_ticket_mask,
    magnitude_pruning_curve,
    parameter_redundancy_analysis,
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


# ─── Fisher Information Importance ─────────────────────────────────────────


class TestFisherInformationImportance:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = fisher_information_importance(model, seqs, "blocks.0.mlp.W_in")
        assert "importance" in result
        assert "mean_importance" in result
        assert "top_fraction_90" in result

    def test_importance_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = fisher_information_importance(model, seqs, "blocks.0.mlp.W_in")
        W = model.blocks[0].mlp.W_in
        assert result["importance"].shape == W.shape

    def test_nonnegative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = fisher_information_importance(model, seqs, "blocks.0.mlp.W_in")
        assert np.all(result["importance"] >= -1e-10)

    def test_top_fraction_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = fisher_information_importance(model, seqs, "blocks.0.mlp.W_in")
        assert 0 <= result["top_fraction_90"] <= 1.0


# ─── Activation Variance Importance ────────────────────────────────────────


class TestActivationVarianceImportance:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = activation_variance_importance(model, seqs, "blocks.0.mlp.W_in")
        assert "importance" in result
        assert "sparsity_ratio" in result

    def test_importance_nonneg(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = activation_variance_importance(model, seqs, "blocks.0.mlp.W_in")
        assert np.all(result["importance"] >= 0)

    def test_sparsity_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = activation_variance_importance(model, seqs, "blocks.0.mlp.W_in")
        assert 0 <= result["sparsity_ratio"] <= 1.0


# ─── Lottery Ticket Mask ──────────────────────────────────────────────────


class TestLotteryTicketMask:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = lottery_ticket_mask(model, seqs, "blocks.0.mlp.W_in", target_sparsity=0.5)
        assert "mask" in result
        assert "n_kept" in result
        assert "n_total" in result

    def test_mask_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = lottery_ticket_mask(model, seqs, "blocks.0.mlp.W_in", target_sparsity=0.5)
        W = model.blocks[0].mlp.W_in
        assert result["mask"].shape == W.shape

    def test_sparsity_approximate(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = lottery_ticket_mask(model, seqs, "blocks.0.mlp.W_in", target_sparsity=0.5)
        # Should be approximately 50% sparse
        assert 0.3 <= result["actual_sparsity"] <= 0.7

    def test_mask_is_boolean(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = lottery_ticket_mask(model, seqs, "blocks.0.mlp.W_in")
        assert result["mask"].dtype == bool

    def test_n_kept_consistent(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = lottery_ticket_mask(model, seqs, "blocks.0.mlp.W_in")
        assert result["n_kept"] == int(np.sum(result["mask"]))


# ─── Magnitude Pruning Curve ──────────────────────────────────────────────


class TestMagnitudePruningCurve:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = magnitude_pruning_curve(model, seqs, "blocks.0.mlp.W_in",
                                          _metric, sparsity_levels=[0.0, 0.5])
        assert "sparsity_levels" in result
        assert "metrics" in result

    def test_metric_count_matches(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        levels = [0.0, 0.3, 0.6, 0.9]
        result = magnitude_pruning_curve(model, seqs, "blocks.0.mlp.W_in",
                                          _metric, sparsity_levels=levels)
        assert len(result["metrics"]) == len(levels)

    def test_zero_sparsity_matches_clean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        seqs = [tokens]
        clean_value = _metric(model(tokens))
        result = magnitude_pruning_curve(model, seqs, "blocks.0.mlp.W_in",
                                          _metric, sparsity_levels=[0.0])
        assert abs(result["metrics"][0] - clean_value) < 1e-5


# ─── Parameter Redundancy Analysis ────────────────────────────────────────


class TestParameterRedundancyAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        paths = ["blocks.0.mlp.W_in", "blocks.1.mlp.W_in"]
        result = parameter_redundancy_analysis(model, paths)
        assert "redundancy_scores" in result
        assert "similarity_matrices" in result
        assert "most_redundant" in result

    def test_scores_in_range(self):
        model = _make_model()
        paths = ["blocks.0.mlp.W_in"]
        result = parameter_redundancy_analysis(model, paths)
        for score in result["redundancy_scores"].values():
            assert 0 <= score <= 1.0

    def test_similarity_matrix_shape(self):
        model = _make_model()
        paths = ["blocks.0.mlp.W_in"]
        result = parameter_redundancy_analysis(model, paths)
        sim = result["similarity_matrices"]["blocks.0.mlp.W_in"]
        W = model.blocks[0].mlp.W_in
        assert sim.shape[0] == W.shape[0]

    def test_most_redundant_is_valid(self):
        model = _make_model()
        paths = ["blocks.0.mlp.W_in", "blocks.1.mlp.W_in"]
        result = parameter_redundancy_analysis(model, paths)
        assert result["most_redundant"] in paths
