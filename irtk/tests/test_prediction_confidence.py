"""Tests for prediction_confidence module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_confidence import (
    confidence_profile,
    layerwise_confidence_evolution,
    confidence_source_attribution,
    entropy_decomposition,
    confidence_calibration,
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


class TestConfidenceProfile:
    def test_basic(self, model, tokens):
        result = confidence_profile(model, tokens)
        assert "top1_probs" in result
        assert "top1_tokens" in result
        assert "entropy" in result
        assert "confidence_mean" in result

    def test_shapes(self, model, tokens):
        result = confidence_profile(model, tokens)
        seq_len = len(tokens)
        assert result["top1_probs"].shape == (seq_len,)
        assert result["top1_tokens"].shape == (seq_len,)
        assert result["entropy"].shape == (seq_len,)

    def test_prob_range(self, model, tokens):
        result = confidence_profile(model, tokens)
        assert np.all(result["top1_probs"] >= 0)
        assert np.all(result["top1_probs"] <= 1.01)

    def test_entropy_nonneg(self, model, tokens):
        result = confidence_profile(model, tokens)
        assert np.all(result["entropy"] >= 0)


class TestLayerwiseConfidenceEvolution:
    def test_basic(self, model, tokens):
        result = layerwise_confidence_evolution(model, tokens)
        assert "layer_entropy" in result
        assert "layer_top1_prob" in result
        assert "confidence_emergence_layer" in result
        assert "entropy_reduction" in result

    def test_shapes(self, model, tokens):
        result = layerwise_confidence_evolution(model, tokens)
        nl = model.cfg.n_layers
        assert result["layer_entropy"].shape == (nl,)
        assert result["layer_top1_prob"].shape == (nl,)
        assert result["entropy_reduction"].shape == (nl - 1,)

    def test_emergence_valid(self, model, tokens):
        result = layerwise_confidence_evolution(model, tokens)
        assert 0 <= result["confidence_emergence_layer"] < model.cfg.n_layers


class TestConfidenceSourceAttribution:
    def test_basic(self, model, tokens):
        result = confidence_source_attribution(model, tokens)
        assert "component_confidence_effects" in result
        assert "confidence_boosters" in result
        assert "confidence_reducers" in result
        assert "top_token_attributions" in result

    def test_all_components(self, model, tokens):
        result = confidence_source_attribution(model, tokens)
        nl = model.cfg.n_layers
        assert len(result["component_confidence_effects"]) == nl * 2  # attn + mlp


class TestEntropyDecomposition:
    def test_basic(self, model, tokens):
        result = entropy_decomposition(model, tokens)
        assert "total_entropy" in result
        assert "component_entropy_contributions" in result
        assert "entropy_from_embedding" in result
        assert "entropy_from_attention" in result
        assert "entropy_from_mlp" in result

    def test_entropy_nonneg(self, model, tokens):
        result = entropy_decomposition(model, tokens)
        assert result["total_entropy"] >= 0
        assert result["entropy_from_embedding"] >= 0
        assert result["entropy_from_attention"] >= 0
        assert result["entropy_from_mlp"] >= 0


class TestConfidenceCalibration:
    def test_basic(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5]), jnp.array([10, 20, 30, 40, 49])]
        correct_tokens = [0, 0, 0]  # doesn't matter for structure
        result = confidence_calibration(model, tokens_list, correct_tokens)
        assert "confidences" in result
        assert "correct" in result
        assert "mean_confidence" in result
        assert "accuracy" in result
        assert "calibration_error" in result

    def test_lengths_match(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5])]
        correct_tokens = [0, 0]
        result = confidence_calibration(model, tokens_list, correct_tokens)
        assert len(result["confidences"]) == 2
        assert len(result["correct"]) == 2

    def test_ranges(self, model, tokens):
        tokens_list = [tokens]
        correct_tokens = [0]
        result = confidence_calibration(model, tokens_list, correct_tokens)
        assert 0 <= result["mean_confidence"] <= 1
        assert 0 <= result["accuracy"] <= 1
        assert result["calibration_error"] >= 0
