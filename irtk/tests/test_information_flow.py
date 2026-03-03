"""Tests for information-theoretic layer analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.information_flow import (
    layer_entropy,
    mutual_information_estimate,
    compression_analysis,
    information_flow_by_position,
    information_bottleneck_curve,
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


# ─── Layer Entropy ────────────────────────────────────────────────────────────


class TestLayerEntropy:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_entropy(model, seqs)
        assert "layer_entropies" in result
        assert "embedding_entropy" in result
        assert "entropy_trend" in result

    def test_entropies_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_entropy(model, seqs)
        assert len(result["layer_entropies"]) == 2

    def test_entropies_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_entropy(model, seqs)
        assert np.all(result["layer_entropies"] >= 0)

    def test_trend_valid(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_entropy(model, seqs)
        assert result["entropy_trend"] in ("increasing", "decreasing", "non-monotonic", "flat")


# ─── Mutual Information Estimate ──────────────────────────────────────────────


class TestMutualInformationEstimate:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = mutual_information_estimate(model, seqs, layer=0)
        assert "mutual_information" in result
        assert "input_entropy" in result
        assert "normalized_mi" in result

    def test_mi_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = mutual_information_estimate(model, seqs, layer=0)
        assert result["mutual_information"] >= 0

    def test_normalized_bounded(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = mutual_information_estimate(model, seqs, layer=0)
        assert result["normalized_mi"] >= 0


# ─── Compression Analysis ────────────────────────────────────────────────────


class TestCompressionAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = compression_analysis(model, seqs)
        assert "effective_dimensions" in result
        assert "compression_ratios" in result
        assert "compression_phase" in result

    def test_dimensions_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = compression_analysis(model, seqs)
        assert len(result["effective_dimensions"]) == 2

    def test_ratios_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = compression_analysis(model, seqs)
        assert np.all(result["compression_ratios"] >= 0)


# ─── Information Flow by Position ─────────────────────────────────────────────


class TestInformationFlowByPosition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = information_flow_by_position(model, tokens, source_pos=0, target_pos=3)
        assert "attention_to_source" in result
        assert "layer_influence" in result
        assert "peak_layer" in result

    def test_attention_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = information_flow_by_position(model, tokens, source_pos=0, target_pos=3)
        assert result["attention_to_source"].shape == (2, 4)

    def test_peak_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = information_flow_by_position(model, tokens, source_pos=0, target_pos=3)
        assert 0 <= result["peak_layer"] < 2


# ─── Information Bottleneck Curve ─────────────────────────────────────────────


class TestInformationBottleneckCurve:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = information_bottleneck_curve(model, seqs, _metric)
        assert "compression" in result
        assert "prediction" in result
        assert "ib_trade_off" in result
        assert "optimal_layer" in result

    def test_arrays_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = information_bottleneck_curve(model, seqs, _metric)
        assert len(result["compression"]) == 2
        assert len(result["prediction"]) == 2

    def test_optimal_valid(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = information_bottleneck_curve(model, seqs, _metric)
        assert 0 <= result["optimal_layer"] < 2
