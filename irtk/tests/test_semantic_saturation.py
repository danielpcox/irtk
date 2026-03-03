"""Tests for semantic saturation analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.semantic_saturation import (
    semantic_information_saturation,
    redundant_layer_detection,
    token_saturation_curve,
    representation_stabilization_point,
    early_vs_late_computation_balance,
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


class TestSemanticInformationSaturation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = semantic_information_saturation(model, tokens, target_token=5)
        assert "layer_probs" in result
        assert "saturation_layer" in result
        assert "information_gain" in result
        assert "final_prob" in result

    def test_probs_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = semantic_information_saturation(model, tokens, target_token=5)
        assert len(result["layer_probs"]) == 2

    def test_final_prob_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = semantic_information_saturation(model, tokens, target_token=5)
        assert 0 <= result["final_prob"] <= 1.0


class TestRedundantLayerDetection:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = redundant_layer_detection(model, tokens, _metric)
        assert "layer_effects" in result
        assert "redundant_layers" in result
        assert "essential_layers" in result

    def test_all_layers_classified(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = redundant_layer_detection(model, tokens, _metric)
        total = len(result["redundant_layers"]) + len(result["essential_layers"])
        assert total == 2


class TestTokenSaturationCurve:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_saturation_curve(model, tokens)
        assert "layer_entropies" in result
        assert "entropy_reduction" in result
        assert "steepest_drop_layer" in result

    def test_entropies_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_saturation_curve(model, tokens)
        assert len(result["layer_entropies"]) == 2


class TestRepresentationStabilizationPoint:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = representation_stabilization_point(model, tokens)
        assert "layer_distances" in result
        assert "stabilization_layer" in result
        assert "total_drift" in result

    def test_distances_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = representation_stabilization_point(model, tokens)
        assert len(result["layer_distances"]) == 1  # n_layers - 1


class TestEarlyVsLateComputationBalance:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = early_vs_late_computation_balance(model, tokens, _metric)
        assert "early_effect" in result
        assert "late_effect" in result
        assert "balance_ratio" in result

    def test_balance_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = early_vs_late_computation_balance(model, tokens, _metric)
        assert 0.0 <= result["balance_ratio"] <= 1.0
