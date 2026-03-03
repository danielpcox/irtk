"""Tests for sequence dynamics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sequence_dynamics import (
    repetition_handling_analysis,
    long_range_dependency_tracking,
    position_bias_strength,
    length_effect_on_circuits,
    boundary_effect_analysis,
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


class TestRepetitionHandlingAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 0, 1, 2])  # Contains repetitions
        result = repetition_handling_analysis(model, tokens)
        assert "first_occurrence_entropy" in result
        assert "repeated_occurrence_entropy" in result
        assert "entropy_reduction" in result
        assert "induction_score" in result

    def test_repeated_tokens_counted(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 0, 1, 2])
        result = repetition_handling_analysis(model, tokens)
        assert result["n_repeated_tokens"] == 3  # 0, 1, 2 each repeated

    def test_no_repetitions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = repetition_handling_analysis(model, tokens)
        assert result["n_repeated_tokens"] == 0


class TestLongRangeDependencyTracking:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = long_range_dependency_tracking(model, tokens, source_pos=0, target_pos=-1)
        assert "direct_attention_per_layer" in result
        assert "source_ablation_effect" in result
        assert "information_retention" in result

    def test_attention_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = long_range_dependency_tracking(model, tokens)
        assert len(result["direct_attention_per_layer"]) == 2

    def test_ablation_effect_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = long_range_dependency_tracking(model, tokens)
        assert result["source_ablation_effect"] >= 0


class TestPositionBiasStrength:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        result = position_bias_strength(model, tokens)
        assert "recency_bias" in result
        assert "primacy_bias" in result
        assert "distance_decay_rate" in result
        assert "position_entropy" in result

    def test_biases_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        result = position_bias_strength(model, tokens)
        assert 0.0 <= result["recency_bias"] <= 1.0
        assert 0.0 <= result["primacy_bias"] <= 1.0


class TestLengthEffectOnCircuits:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        result = length_effect_on_circuits(model, tokens, _metric)
        assert "lengths" in result
        assert "metrics" in result
        assert "length_sensitivity" in result

    def test_metrics_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
        result = length_effect_on_circuits(model, tokens, _metric, lengths=[2, 4, 8])
        assert len(result["metrics"]) == 3


class TestBoundaryEffectAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = boundary_effect_analysis(model, tokens)
        assert "start_norm_ratio" in result
        assert "end_norm_ratio" in result
        assert "start_attention_concentration" in result
        assert "end_confidence_boost" in result
        assert "boundary_effect_strength" in result

    def test_ratios_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = boundary_effect_analysis(model, tokens)
        assert result["start_norm_ratio"] > 0
        assert result["end_norm_ratio"] > 0
