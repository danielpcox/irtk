"""Tests for representation engineering tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.representation_engineering import (
    extract_reading_vectors,
    reading_vector_scan,
    control_vector_intervention,
    representation_score,
    concept_suppression_curve,
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


# ─── Extract Reading Vectors ─────────────────────────────────────────────


class TestExtractReadingVectors:
    def test_returns_dict(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        neg = [jnp.array([10, 11, 12, 13]), jnp.array([14, 15, 16, 17])]
        result = extract_reading_vectors(model, pos, neg, "blocks.0.hook_resid_post")
        assert "reading_vectors" in result
        assert "explained_variance" in result
        assert "separation_score" in result

    def test_vector_shape(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3])]
        neg = [jnp.array([10, 11, 12, 13])]
        result = extract_reading_vectors(model, pos, neg, "blocks.0.hook_resid_post", n_components=2)
        assert result["reading_vectors"].shape == (2, 16)

    def test_separation_in_range(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        neg = [jnp.array([10, 11, 12, 13]), jnp.array([14, 15, 16, 17])]
        result = extract_reading_vectors(model, pos, neg, "blocks.0.hook_resid_post")
        assert -1.0 <= result["separation_score"] <= 1.0

    def test_empty_prompts(self):
        model = _make_model()
        result = extract_reading_vectors(model, [], [], "blocks.0.hook_resid_post")
        assert result["separation_score"] == 0.0


# ─── Reading Vector Scan ─────────────────────────────────────────────────


class TestReadingVectorScan:
    def test_returns_dict(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3])]
        neg = [jnp.array([10, 11, 12, 13])]
        result = reading_vector_scan(model, pos, neg)
        assert "reading_vectors" in result
        assert "separation_scores" in result
        assert "best_layer" in result

    def test_vectors_shape(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3])]
        neg = [jnp.array([10, 11, 12, 13])]
        result = reading_vector_scan(model, pos, neg)
        assert result["reading_vectors"].shape == (2, 16)  # n_layers, d_model

    def test_best_layer_valid(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3])]
        neg = [jnp.array([10, 11, 12, 13])]
        result = reading_vector_scan(model, pos, neg)
        assert 0 <= result["best_layer"] < 2

    def test_scores_length(self):
        model = _make_model()
        pos = [jnp.array([0, 1, 2, 3])]
        neg = [jnp.array([10, 11, 12, 13])]
        result = reading_vector_scan(model, pos, neg)
        assert len(result["separation_scores"]) == 2


# ─── Control Vector Intervention ──────────────────────────────────────────


class TestControlVectorIntervention:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = control_vector_intervention(model, tokens, vecs, coefficient=1.0)
        assert "original_logits" in result
        assert "steered_logits" in result
        assert "logit_diff" in result

    def test_logit_diff_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = control_vector_intervention(model, tokens, vecs)
        assert len(result["logit_diff"]) == model.cfg.d_vocab

    def test_top_changed_tokens(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = control_vector_intervention(model, tokens, vecs)
        assert len(result["top_changed_tokens"]) <= 20


# ─── Representation Score ────────────────────────────────────────────────


class TestRepresentationScore:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rv = np.random.randn(16)
        result = representation_score(model, tokens, rv, "blocks.0.hook_resid_post")
        assert "scores" in result
        assert "mean_score" in result
        assert "max_pos" in result

    def test_scores_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rv = np.random.randn(16)
        result = representation_score(model, tokens, rv, "blocks.0.hook_resid_post")
        assert len(result["scores"]) == 4

    def test_max_pos_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rv = np.random.randn(16)
        result = representation_score(model, tokens, rv, "blocks.0.hook_resid_post")
        assert 0 <= result["max_pos"] < 4

    def test_bad_hook(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rv = np.random.randn(16)
        result = representation_score(model, tokens, rv, "nonexistent")
        assert len(result["scores"]) == 0


# ─── Concept Suppression Curve ────────────────────────────────────────────


class TestConceptSuppressionCurve:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = concept_suppression_curve(model, tokens, vecs, _metric)
        assert "coefficients" in result
        assert "metrics" in result
        assert "baseline_metric" in result
        assert "sensitivity" in result

    def test_lengths_match(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = concept_suppression_curve(model, tokens, vecs, _metric)
        assert len(result["coefficients"]) == len(result["metrics"])

    def test_custom_coefficients(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = concept_suppression_curve(model, tokens, vecs, _metric, coefficients=[-1.0, 0.0, 1.0])
        assert len(result["metrics"]) == 3

    def test_sensitivity_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vecs = {"blocks.0.hook_resid_post": np.random.randn(16) * 0.1}
        result = concept_suppression_curve(model, tokens, vecs, _metric)
        assert result["sensitivity"] >= 0
