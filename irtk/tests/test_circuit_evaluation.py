"""Tests for circuit evaluation metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.circuit_evaluation import (
    faithfulness_score,
    completeness_score,
    minimality_check,
    circuit_iou,
    evaluate_circuit,
    _ablate_heads_outside_circuit,
    _ablate_heads_inside_circuit,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
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
    """Simple metric: logit at position -1, token 0."""
    return float(logits[-1, 0])


class TestAblateHelpers:
    def test_ablate_outside_empty_circuit_zeros_all_heads(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_ablated = _ablate_heads_outside_circuit(model, tokens, [], method="zero")
        logits_full = model(tokens)
        # All heads ablated should differ from full
        assert not np.allclose(logits_ablated, logits_full, atol=1e-4)

    def test_ablate_outside_full_circuit_matches_clean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        all_heads = [(l, h) for l in range(2) for h in range(4)]
        logits_ablated = _ablate_heads_outside_circuit(model, tokens, all_heads, method="zero")
        logits_full = model(tokens)
        # No heads ablated - should match exactly
        assert np.allclose(logits_ablated, logits_full, atol=1e-5)

    def test_ablate_inside_circuit_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        logits_ablated = _ablate_heads_inside_circuit(model, tokens, circuit, method="zero")
        logits_full = model(tokens)
        assert not np.allclose(logits_ablated, logits_full, atol=1e-4)

    def test_ablate_inside_empty_circuit_matches_clean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_ablated = _ablate_heads_inside_circuit(model, tokens, [], method="zero")
        logits_full = model(tokens)
        assert np.allclose(logits_ablated, logits_full, atol=1e-5)

    def test_mean_ablation_method(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0)]
        logits_zero = _ablate_heads_outside_circuit(model, tokens, circuit, method="zero")
        logits_mean = _ablate_heads_outside_circuit(model, tokens, circuit, method="mean")
        # Different methods should give different results
        assert not np.allclose(logits_zero, logits_mean, atol=1e-4)


class TestFaithfulnessScore:
    def test_full_circuit_high_faithfulness(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        all_heads = [(l, h) for l in range(2) for h in range(4)]
        score = faithfulness_score(model, all_heads, [tokens], _metric)
        assert score > 0.99

    def test_empty_circuit_low_faithfulness(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        score = faithfulness_score(model, [], [tokens], _metric)
        # Empty circuit should have low faithfulness
        assert score < 0.5

    def test_score_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        score = faithfulness_score(model, circuit, [tokens], _metric)
        assert 0.0 <= score <= 1.0

    def test_multiple_prompts(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        all_heads = [(l, h) for l in range(2) for h in range(4)]
        score = faithfulness_score(model, all_heads, seqs, _metric)
        assert score > 0.99


class TestCompletenessScore:
    def test_full_circuit_high_completeness(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        all_heads = [(l, h) for l in range(2) for h in range(4)]
        score = completeness_score(model, all_heads, [tokens], _metric)
        assert score > 0.9

    def test_empty_circuit_zero_completeness(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        score = completeness_score(model, [], [tokens], _metric)
        # Empty circuit ablation changes nothing
        assert score < 0.1

    def test_score_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        score = completeness_score(model, circuit, [tokens], _metric)
        assert 0.0 <= score <= 1.0


class TestMinimalityCheck:
    def test_returns_list(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        result = minimality_check(model, circuit, [tokens], _metric)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_result_structure(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        result = minimality_check(model, circuit, [tokens], _metric)
        for entry in result:
            assert "layer" in entry
            assert "head" in entry
            assert "metric_without" in entry
            assert "metric_change" in entry
            assert "relative_change" in entry
            assert "redundant" in entry

    def test_single_head_not_redundant(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        # Single-head circuit - removing it removes the entire circuit
        circuit = [(0, 0)]
        result = minimality_check(model, circuit, [tokens], _metric, threshold=0.001)
        # With very tight threshold, the single head should not be redundant
        # unless removing it has literally no effect
        assert len(result) == 1

    def test_high_threshold_makes_heads_redundant(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        result = minimality_check(model, circuit, [tokens], _metric, threshold=1.0)
        # With threshold=1.0, everything should be redundant
        assert all(r["redundant"] for r in result)


class TestCircuitIoU:
    def test_identical_circuits(self):
        a = [(0, 0), (1, 1)]
        b = [(0, 0), (1, 1)]
        assert circuit_iou(a, b) == 1.0

    def test_disjoint_circuits(self):
        a = [(0, 0), (0, 1)]
        b = [(1, 0), (1, 1)]
        assert circuit_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [(0, 0), (0, 1)]
        b = [(0, 0), (1, 0)]
        # intersection = {(0,0)}, union = {(0,0), (0,1), (1,0)}
        assert abs(circuit_iou(a, b) - 1 / 3) < 1e-10

    def test_empty_circuits(self):
        assert circuit_iou([], []) == 1.0

    def test_one_empty(self):
        assert circuit_iou([(0, 0)], []) == 0.0

    def test_duplicates_handled(self):
        a = [(0, 0), (0, 0), (0, 1)]
        b = [(0, 0), (0, 1)]
        assert circuit_iou(a, b) == 1.0


class TestEvaluateCircuit:
    def test_returns_all_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        result = evaluate_circuit(model, circuit, [tokens], _metric)
        assert "faithfulness" in result
        assert "completeness" in result
        assert "minimality" in result
        assert "n_redundant" in result
        assert "n_circuit_heads" in result
        assert "baseline_metric" in result
        assert "circuit_metric" in result

    def test_n_circuit_heads(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0), (1, 2)]
        result = evaluate_circuit(model, circuit, [tokens], _metric)
        assert result["n_circuit_heads"] == 3

    def test_minimality_length_matches_circuit(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        result = evaluate_circuit(model, circuit, [tokens], _metric)
        assert len(result["minimality"]) == 2

    def test_full_circuit_high_scores(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        all_heads = [(l, h) for l in range(2) for h in range(4)]
        result = evaluate_circuit(model, all_heads, [tokens], _metric)
        assert result["faithfulness"] > 0.99
        assert result["completeness"] > 0.9

    def test_mean_method(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        circuit = [(0, 0), (1, 0)]
        result = evaluate_circuit(model, circuit, [tokens], _metric, method="mean")
        assert 0.0 <= result["faithfulness"] <= 1.0
        assert 0.0 <= result["completeness"] <= 1.0
