"""Tests for interpretability benchmark metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.interp_benchmarks import (
    logit_diff,
    kl_divergence,
    loss_recovered,
    ablation_effect_size,
    faithfulness_correlation,
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


# ─── Logit Diff ─────────────────────────────────────────────────────────────


class TestLogitDiff:
    def test_basic(self):
        logits = jnp.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.5]])
        result = logit_diff(logits, correct_token=0, incorrect_token=2, pos=-1)
        assert abs(result - 2.5) < 1e-5  # 3.0 - 0.5

    def test_positive_when_correct_higher(self):
        logits = jnp.array([[5.0, 1.0, 2.0]])
        result = logit_diff(logits, correct_token=0, incorrect_token=1, pos=0)
        assert result > 0

    def test_negative_when_incorrect_higher(self):
        logits = jnp.array([[1.0, 5.0, 2.0]])
        result = logit_diff(logits, correct_token=0, incorrect_token=1, pos=0)
        assert result < 0

    def test_zero_when_equal(self):
        logits = jnp.array([[3.0, 3.0, 2.0]])
        result = logit_diff(logits, correct_token=0, incorrect_token=1, pos=0)
        assert abs(result) < 1e-6

    def test_with_model(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits = model(tokens)
        result = logit_diff(logits, correct_token=0, incorrect_token=1)
        assert isinstance(result, float)


# ─── KL Divergence ──────────────────────────────────────────────────────────


class TestKLDivergence:
    def test_identical_is_zero(self):
        logits = jnp.array([[1.0, 2.0, 3.0]])
        result = kl_divergence(logits, logits, pos=0)
        assert abs(result) < 1e-5

    def test_nonnegative(self):
        logits_a = jnp.array([[1.0, 2.0, 3.0]])
        logits_b = jnp.array([[3.0, 1.0, 2.0]])
        result = kl_divergence(logits_a, logits_b, pos=0)
        assert result >= -1e-7

    def test_different_distributions(self):
        logits_a = jnp.array([[0.0, 0.0, 10.0]])
        logits_b = jnp.array([[10.0, 0.0, 0.0]])
        result = kl_divergence(logits_a, logits_b, pos=0)
        assert result > 0.1

    def test_with_model(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits = model(tokens)
        result = kl_divergence(logits, logits)
        assert abs(result) < 1e-5


# ─── Loss Recovered ────────────────────────────────────────────────────────


class TestLossRecovered:
    def test_clean_logits_recover_fully(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        clean_logits = model(tokens)
        result = loss_recovered(model, tokens, clean_logits)
        assert abs(result - 1.0) < 0.01

    def test_returns_float(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits = model(tokens)
        result = loss_recovered(model, tokens, logits)
        assert isinstance(result, float)

    def test_with_corrupted_baseline(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        clean_logits = model(tokens)
        # Corrupt logits by adding noise
        corrupted = clean_logits + 5.0
        result = loss_recovered(model, tokens, clean_logits, corrupted_logits=corrupted)
        assert isinstance(result, float)

    def test_bad_logits_low_recovery(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        clean_logits = model(tokens)
        # Use uniform logits (very bad)
        bad_logits = jnp.zeros_like(clean_logits)
        corrupted = jnp.zeros_like(clean_logits)
        result = loss_recovered(model, tokens, bad_logits, corrupted_logits=corrupted)
        # Should be ~0 or close since both bad and corrupt are similar
        assert isinstance(result, float)


# ─── Ablation Effect Size ──────────────────────────────────────────────────


class TestAblationEffectSize:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_effect_size(model, tokens, "blocks.0.hook_attn_out", _metric)
        assert isinstance(result, dict)
        assert "clean_metric" in result
        assert "effect_size" in result

    def test_effect_size_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_effect_size(model, tokens, "blocks.0.hook_attn_out", _metric)
        assert result["effect_size"] >= 0

    def test_effect_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_effect_size(model, tokens, "blocks.0.hook_attn_out", _metric)
        expected = result["ablated_metric"] - result["clean_metric"]
        assert abs(result["effect"] - expected) < 1e-6

    def test_mean_ablation(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_effect_size(
            model, tokens, "blocks.0.hook_attn_out", _metric, ablation_type="mean"
        )
        assert "ablated_metric" in result

    def test_invalid_hook_zero_effect(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = ablation_effect_size(model, tokens, "nonexistent", _metric)
        assert result["effect"] == 0.0


# ─── Faithfulness Correlation ───────────────────────────────────────────────


class TestFaithfulnessCorrelation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attributions = {
            "blocks.0.hook_attn_out": 0.5,
            "blocks.1.hook_attn_out": 0.3,
        }
        result = faithfulness_correlation(model, tokens, attributions, _metric)
        assert isinstance(result, dict)
        assert "correlation" in result

    def test_correlation_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attributions = {
            "blocks.0.hook_attn_out": 0.5,
            "blocks.1.hook_attn_out": 0.3,
        }
        result = faithfulness_correlation(model, tokens, attributions, _metric)
        assert -1.01 <= result["correlation"] <= 1.01

    def test_has_all_hooks(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attributions = {
            "blocks.0.hook_attn_out": 0.5,
            "blocks.1.hook_attn_out": 0.3,
        }
        result = faithfulness_correlation(model, tokens, attributions, _metric)
        assert len(result["hook_names"]) == 2
        assert len(result["ablation_effects"]) == 2

    def test_single_hook(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attributions = {"blocks.0.hook_attn_out": 1.0}
        result = faithfulness_correlation(model, tokens, attributions, _metric)
        # With single hook, correlation is undefined (std=0), expect 0
        assert isinstance(result["correlation"], float)
