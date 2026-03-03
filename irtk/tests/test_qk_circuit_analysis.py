"""Tests for qk_circuit_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.qk_circuit_analysis import (
    qk_eigenvalue_structure,
    positional_vs_content_qk,
    qk_pattern_prediction,
    qk_composition_analysis,
    effective_receptive_field,
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


class TestQKEigenvalueStructure:
    def test_basic(self, model):
        result = qk_eigenvalue_structure(model, layer=0, head=0)
        assert "eigenvalues" in result
        assert "qk_trace" in result
        assert "qk_rank" in result
        assert "positive_negative_ratio" in result

    def test_eigenvalue_count(self, model):
        result = qk_eigenvalue_structure(model, layer=0, head=0, top_k=3)
        assert len(result["eigenvalues"]) == 3

    def test_rank_positive(self, model):
        result = qk_eigenvalue_structure(model, layer=0, head=0)
        assert result["qk_rank"] >= 0


class TestPositionalVsContentQK:
    def test_basic(self, model, tokens):
        result = positional_vs_content_qk(model, tokens, layer=0, head=0)
        assert "content_scores" in result
        assert "positional_scores" in result
        assert "content_fraction" in result
        assert "positional_fraction" in result

    def test_shapes(self, model, tokens):
        seq_len = len(tokens)
        result = positional_vs_content_qk(model, tokens, layer=0, head=0)
        assert result["content_scores"].shape == (seq_len, seq_len)
        assert result["positional_scores"].shape == (seq_len, seq_len)

    def test_fractions_sum(self, model, tokens):
        result = positional_vs_content_qk(model, tokens, layer=0, head=0)
        total = result["content_fraction"] + result["positional_fraction"]
        assert abs(total - 1.0) < 0.01


class TestQKPatternPrediction:
    def test_basic(self, model, tokens):
        result = qk_pattern_prediction(model, tokens, layer=0, head=0)
        assert "predicted_pattern" in result
        assert "actual_pattern" in result
        assert "correlation" in result
        assert "mse" in result

    def test_shapes(self, model, tokens):
        seq_len = len(tokens)
        result = qk_pattern_prediction(model, tokens, layer=0, head=0)
        assert result["predicted_pattern"].shape == (seq_len, seq_len)
        assert result["actual_pattern"].shape == (seq_len, seq_len)

    def test_correlation_range(self, model, tokens):
        result = qk_pattern_prediction(model, tokens, layer=0, head=0)
        assert -1.01 <= result["correlation"] <= 1.01


class TestQKCompositionAnalysis:
    def test_basic(self, model):
        result = qk_composition_analysis(model, 0, 0, 1, 0)
        assert "q_composition_score" in result
        assert "k_composition_score" in result
        assert "composed_qk_rank" in result
        assert "composition_strength" in result

    def test_scores_nonneg(self, model):
        result = qk_composition_analysis(model, 0, 0, 1, 0)
        assert result["q_composition_score"] >= 0
        assert result["k_composition_score"] >= 0
        assert result["composition_strength"] >= 0


class TestEffectiveReceptiveField:
    def test_basic(self, model, tokens):
        result = effective_receptive_field(model, tokens, layer=0, head=0)
        assert "attention_pattern" in result
        assert "mean_attention_distance" in result
        assert "receptive_field_width" in result
        assert "peak_positions" in result
        assert "attention_entropy" in result

    def test_shapes(self, model, tokens):
        seq_len = len(tokens)
        result = effective_receptive_field(model, tokens, layer=0, head=0)
        assert result["attention_pattern"].shape == (seq_len, seq_len)
        assert result["peak_positions"].shape == (seq_len,)
        assert result["attention_entropy"].shape == (seq_len,)

    def test_distance_nonneg(self, model, tokens):
        result = effective_receptive_field(model, tokens, layer=0, head=0)
        assert result["mean_attention_distance"] >= 0
