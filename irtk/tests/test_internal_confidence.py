"""Tests for internal confidence analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.internal_confidence import (
    confidence_direction,
    internal_vs_output_confidence_gap,
    confidence_accumulation_profile,
    uncertainty_decomposition,
    self_consistency_probe,
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


# ─── Confidence Direction ────────────────────────────────────────────────────


class TestConfidenceDirection:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = confidence_direction(
            model, tokens, correct_token=5, hook_name="blocks.0.hook_resid_post"
        )
        assert "direction" in result
        assert "confidence_score" in result
        assert "output_prob" in result

    def test_direction_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = confidence_direction(
            model, tokens, correct_token=5, hook_name="blocks.0.hook_resid_post"
        )
        assert result["direction"].shape == (16,)

    def test_output_prob_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = confidence_direction(
            model, tokens, correct_token=5, hook_name="blocks.0.hook_resid_post"
        )
        assert 0.0 <= result["output_prob"] <= 1.0


# ─── Internal vs Output Confidence Gap ──────────────────────────────────────


class TestInternalVsOutputConfidenceGap:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = internal_vs_output_confidence_gap(model, tokens, correct_token=5)
        assert "layer_internal_conf" in result
        assert "output_prob" in result
        assert "gaps" in result
        assert "max_gap_layer" in result

    def test_layers_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = internal_vs_output_confidence_gap(model, tokens, correct_token=5)
        assert len(result["layer_internal_conf"]) == 2

    def test_gap_layer_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = internal_vs_output_confidence_gap(model, tokens, correct_token=5)
        assert 0 <= result["max_gap_layer"] < 2


# ─── Confidence Accumulation Profile ────────────────────────────────────────


class TestConfidenceAccumulationProfile:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([5, 6, 7, 8])]
        result = confidence_accumulation_profile(model, seqs, [5, 10])
        assert "mean_confidence" in result
        assert "std_confidence" in result
        assert "monotonicity" in result

    def test_confidence_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([5, 6, 7, 8])]
        result = confidence_accumulation_profile(model, seqs, [5, 10])
        assert len(result["mean_confidence"]) == 2

    def test_empty_sequences(self):
        model = _make_model()
        result = confidence_accumulation_profile(model, [], [])
        assert result["monotonicity"] == 0.0


# ─── Uncertainty Decomposition ──────────────────────────────────────────────


class TestUncertaintyDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([5, 6, 7, 8]),
                jnp.array([10, 11, 12, 13])]
        result = uncertainty_decomposition(
            model, seqs, "blocks.0.hook_resid_post", top_k=3
        )
        assert "uncertainty_axes" in result
        assert "explained_variance" in result
        assert "entropy_correlations" in result

    def test_axes_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([5, 6, 7, 8]),
                jnp.array([10, 11, 12, 13])]
        result = uncertainty_decomposition(
            model, seqs, "blocks.0.hook_resid_post", top_k=3
        )
        assert result["uncertainty_axes"].shape == (3, 16)

    def test_empty_sequences(self):
        model = _make_model()
        result = uncertainty_decomposition(model, [], "blocks.0.hook_resid_post")
        assert result["total_variance"] == 0.0


# ─── Self Consistency Probe ──────────────────────────────────────────────────


class TestSelfConsistencyProbe:
    def test_returns_dict(self):
        model = _make_model()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([0, 1, 2, 4])
        result = self_consistency_probe(
            model, tokens_a, tokens_b, "blocks.0.hook_resid_post"
        )
        assert "cosine_similarity" in result
        assert "output_agreement" in result
        assert "kl_divergence" in result
        assert "consistency_score" in result

    def test_same_input_high_consistency(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = self_consistency_probe(
            model, tokens, tokens, "blocks.0.hook_resid_post"
        )
        assert result["cosine_similarity"] > 0.99
        assert result["output_agreement"] is True

    def test_consistency_in_range(self):
        model = _make_model()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([10, 11, 12, 13])
        result = self_consistency_probe(
            model, tokens_a, tokens_b, "blocks.0.hook_resid_post"
        )
        assert 0.0 <= result["consistency_score"] <= 1.1
