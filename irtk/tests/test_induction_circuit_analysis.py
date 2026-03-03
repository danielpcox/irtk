"""Tests for induction_circuit_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.induction_circuit_analysis import (
    induction_circuit_path_tracing,
    matching_score_matrix,
    copy_circuit_verification,
    prefix_search_analysis,
    induction_circuit_completeness,
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
    # Repeated pattern for induction
    return jnp.array([1, 2, 3, 4, 1, 2, 3, 4])


@pytest.fixture
def metric_fn():
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


class TestInductionCircuitPathTracing:
    def test_basic(self, model, tokens):
        result = induction_circuit_path_tracing(model, tokens)
        assert "prev_token_scores" in result
        assert "induction_scores" in result
        assert "circuit_paths" in result
        assert "best_circuit" in result

    def test_shapes(self, model, tokens):
        result = induction_circuit_path_tracing(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["prev_token_scores"].shape == (nl, nh)
        assert result["induction_scores"].shape == (nl, nh)

    def test_scores_nonneg(self, model, tokens):
        result = induction_circuit_path_tracing(model, tokens)
        assert np.all(result["prev_token_scores"] >= 0)
        assert np.all(result["induction_scores"] >= 0)

    def test_best_circuit_tuple(self, model, tokens):
        result = induction_circuit_path_tracing(model, tokens)
        assert len(result["best_circuit"]) == 5


class TestMatchingScoreMatrix:
    def test_basic(self, model, tokens):
        result = matching_score_matrix(model, tokens)
        assert "qk_match_scores" in result
        assert "best_matching_heads" in result
        assert "mean_match_by_layer" in result

    def test_shapes(self, model, tokens):
        result = matching_score_matrix(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["qk_match_scores"].shape == (nl, nh)
        assert result["mean_match_by_layer"].shape == (nl,)

    def test_scores_nonneg(self, model, tokens):
        result = matching_score_matrix(model, tokens)
        assert np.all(result["qk_match_scores"] >= 0)


class TestCopyCircuitVerification:
    def test_basic(self, model, tokens):
        result = copy_circuit_verification(model, tokens, layer=0, head=0)
        assert "copy_score" in result
        assert "token_copy_accuracy" in result
        assert "ov_eigenvalues" in result
        assert "ov_trace" in result

    def test_accuracy_range(self, model, tokens):
        result = copy_circuit_verification(model, tokens, layer=0, head=0)
        assert 0 <= result["token_copy_accuracy"] <= 1

    def test_eigenvalues(self, model, tokens):
        result = copy_circuit_verification(model, tokens, layer=0, head=0)
        assert len(result["ov_eigenvalues"]) > 0


class TestPrefixSearchAnalysis:
    def test_basic(self, model, tokens):
        result = prefix_search_analysis(model, tokens)
        assert "prefix_match_scores" in result
        assert "best_prefix_heads" in result
        assert "prefix_length_sensitivity" in result

    def test_shapes(self, model, tokens):
        result = prefix_search_analysis(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["prefix_match_scores"].shape == (nl, nh)
        assert result["prefix_length_sensitivity"].shape == (nl, nh)


class TestInductionCircuitCompleteness:
    def test_basic(self, model, tokens, metric_fn):
        result = induction_circuit_completeness(model, tokens, metric_fn)
        assert "base_metric" in result
        assert "circuit_heads" in result
        assert "ablation_effects" in result
        assert "circuit_faithfulness" in result
        assert "redundancy_score" in result

    def test_faithfulness_range(self, model, tokens, metric_fn):
        result = induction_circuit_completeness(model, tokens, metric_fn)
        assert result["circuit_faithfulness"] >= 0

    def test_redundancy_range(self, model, tokens, metric_fn):
        result = induction_circuit_completeness(model, tokens, metric_fn)
        assert 0 <= result["redundancy_score"] <= 1
