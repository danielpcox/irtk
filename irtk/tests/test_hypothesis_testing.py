"""Tests for hypothesis_testing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.hypothesis_testing import (
    permutation_test,
    bootstrap_confidence_interval,
    multiple_comparison_correction,
    effect_size_analysis,
    circuit_hypothesis_test,
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
def tokens_a():
    return [jnp.array([0, 5, 10, 15, 20]), jnp.array([1, 6, 11, 16, 21])]


@pytest.fixture
def tokens_b():
    return [jnp.array([25, 30, 35, 40, 45]), jnp.array([26, 31, 36, 41, 46])]


def metric_fn(logits, tokens):
    return jnp.mean(logits[-1])


class TestPermutationTest:
    def test_basic(self, model, tokens_a, tokens_b):
        result = permutation_test(model, tokens_a, tokens_b, metric_fn, n_permutations=50)
        assert "observed_diff" in result
        assert "p_value" in result
        assert "null_distribution" in result
        assert "effect_size" in result

    def test_p_value_range(self, model, tokens_a, tokens_b):
        result = permutation_test(model, tokens_a, tokens_b, metric_fn, n_permutations=50)
        assert 0 <= result["p_value"] <= 1

    def test_null_size(self, model, tokens_a, tokens_b):
        result = permutation_test(model, tokens_a, tokens_b, metric_fn, n_permutations=100)
        assert result["null_distribution"].shape == (100,)


class TestBootstrapCI:
    def test_basic(self, model, tokens_a):
        result = bootstrap_confidence_interval(model, tokens_a, metric_fn, n_bootstrap=50)
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "standard_error" in result

    def test_ci_order(self, model, tokens_a):
        result = bootstrap_confidence_interval(model, tokens_a, metric_fn, n_bootstrap=50)
        assert result["ci_lower"] <= result["ci_upper"]

    def test_bootstrap_size(self, model, tokens_a):
        result = bootstrap_confidence_interval(model, tokens_a, metric_fn, n_bootstrap=200)
        assert result["bootstrap_distribution"].shape == (200,)


class TestMultipleComparison:
    def test_bonferroni(self):
        result = multiple_comparison_correction([0.01, 0.03, 0.5], method="bonferroni")
        assert "corrected_p_values" in result
        assert "significant" in result
        assert result["corrected_p_values"].shape == (3,)

    def test_fdr(self):
        result = multiple_comparison_correction([0.01, 0.03, 0.5], method="fdr")
        assert result["corrected_p_values"].shape == (3,)

    def test_bonferroni_increases(self):
        result = multiple_comparison_correction([0.01, 0.02], method="bonferroni")
        # Bonferroni multiplies by n
        assert float(result["corrected_p_values"][0]) >= 0.01

    def test_bounded(self):
        result = multiple_comparison_correction([0.8, 0.9], method="bonferroni")
        assert float(result["corrected_p_values"][0]) <= 1.0
        assert float(result["corrected_p_values"][1]) <= 1.0


class TestEffectSize:
    def test_basic(self, model, tokens_a):
        def identity_ablation(m):
            return m
        result = effect_size_analysis(model, tokens_a, metric_fn, identity_ablation, n_bootstrap=50)
        assert "mean_effect" in result
        assert "cohens_d" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_identity_near_zero(self, model, tokens_a):
        def identity_ablation(m):
            return m
        result = effect_size_analysis(model, tokens_a, metric_fn, identity_ablation, n_bootstrap=50)
        assert abs(result["mean_effect"]) < 1e-4


class TestCircuitHypothesisTest:
    def test_basic(self, model, tokens_a):
        result = circuit_hypothesis_test(
            model, tokens_a, metric_fn,
            circuit_components=[(0, 0)],
            n_permutations=10,
        )
        assert "circuit_effect" in result
        assert "p_value" in result
        assert "null_effects" in result
        assert "specificity" in result

    def test_p_value_range(self, model, tokens_a):
        result = circuit_hypothesis_test(
            model, tokens_a, metric_fn,
            circuit_components=[(0, 0), (0, 1)],
            n_permutations=10,
        )
        assert 0 <= result["p_value"] <= 1
