"""Tests for multi-model comparison and behavioral alignment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.behavior_alignment import (
    mechanical_correspondence,
    solution_diversity,
    behavioral_alignment_spectrum,
    interpretability_transfer,
    emergence_comparison,
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


# ─── Mechanical Correspondence ────────────────────────────────────────────


class TestMechanicalCorrespondence:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = mechanical_correspondence(model_a, model_b, seqs)
        assert "head_correspondence" in result
        assert "best_matches" in result
        assert "residual_similarity" in result

    def test_correspondence_shape(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = mechanical_correspondence(model_a, model_b, seqs)
        assert result["head_correspondence"].shape == (4, 4)

    def test_same_model_high_similarity(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = mechanical_correspondence(model, model, seqs)
        assert result["mean_correspondence"] > 0.5

    def test_best_matches_length(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = mechanical_correspondence(model_a, model_b, seqs)
        assert len(result["best_matches"]) == 4  # n_heads


# ─── Solution Diversity ──────────────────────────────────────────────────


class TestSolutionDiversity:
    def test_returns_dict(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = solution_diversity(models, seqs, _metric)
        assert "metric_values" in result
        assert "logit_agreement" in result
        assert "diversity_score" in result

    def test_metric_values_shape(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = solution_diversity(models, seqs, _metric)
        assert result["metric_values"].shape == (2, 2)

    def test_diversity_in_range(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = solution_diversity(models, seqs, _metric)
        assert 0 <= result["diversity_score"] <= 1.0

    def test_same_model_low_diversity(self):
        model = _make_model()
        models = [model, model]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = solution_diversity(models, seqs, _metric)
        assert result["diversity_score"] == 0.0


# ─── Behavioral Alignment Spectrum ────────────────────────────────────────


class TestBehavioralAlignmentSpectrum:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = behavioral_alignment_spectrum(model_a, model_b, seqs)
        assert "layer_similarities" in result
        assert "output_similarity" in result
        assert "divergence_layer" in result

    def test_layer_sims_length(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = behavioral_alignment_spectrum(model_a, model_b, seqs)
        assert len(result["layer_similarities"]) == 2

    def test_same_model_high_alignment(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = behavioral_alignment_spectrum(model, model, seqs)
        assert result["output_similarity"] > 0.9


# ─── Interpretability Transfer ────────────────────────────────────────────


class TestInterpretabilityTransfer:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = interpretability_transfer(model_a, model_b, seqs, "blocks.0.hook_resid_post", _metric)
        assert "source_metrics" in result
        assert "target_baseline" in result
        assert "target_transferred" in result
        assert "transfer_success" in result

    def test_arrays_length(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = interpretability_transfer(model_a, model_b, seqs, "blocks.0.hook_resid_post", _metric)
        assert len(result["source_metrics"]) == 2
        assert len(result["target_baseline"]) == 2

    def test_transfer_success_in_range(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = interpretability_transfer(model_a, model_b, seqs, "blocks.0.hook_resid_post", _metric)
        assert 0 <= result["transfer_success"] <= 1.0


# ─── Emergence Comparison ─────────────────────────────────────────────────


class TestEmergenceComparison:
    def test_returns_dict(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = emergence_comparison(models, seqs, "blocks.0.hook_resid_post")
        assert "activation_norms" in result
        assert "sparsity_levels" in result
        assert "feature_overlap" in result
        assert "emergence_order" in result

    def test_norms_shape(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = emergence_comparison(models, seqs, "blocks.0.hook_resid_post")
        assert result["activation_norms"].shape == (2, 2)

    def test_overlap_shape(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = emergence_comparison(models, seqs, "blocks.0.hook_resid_post")
        assert result["feature_overlap"].shape == (2, 2)

    def test_sparsity_in_range(self):
        models = [_make_model(seed=42)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = emergence_comparison(models, seqs, "blocks.0.hook_resid_post")
        assert 0 <= result["sparsity_levels"][0] <= 1.0

    def test_emergence_order_valid(self):
        models = [_make_model(seed=42), _make_model(seed=99)]
        seqs = [jnp.array([0, 1, 2, 3])]
        result = emergence_comparison(models, seqs, "blocks.0.hook_resid_post")
        assert 0 <= result["emergence_order"] < 2
