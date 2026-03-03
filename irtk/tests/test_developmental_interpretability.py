"""Tests for developmental interpretability analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.developmental_interpretability import (
    detect_phase_transitions,
    track_circuit_formation,
    measure_representation_crystallization,
    compare_learning_order,
    grokking_detector,
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


class TestDetectPhaseTransitions:
    def test_returns_dict(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_phase_transitions(checkpoints, tokens)
        assert "similarity_curve" in result
        assert "transition_points" in result
        assert "smoothness" in result

    def test_similarity_curve_length(self):
        checkpoints = [_make_model(s) for s in range(4)]
        tokens = jnp.array([0, 1, 2, 3])
        result = detect_phase_transitions(checkpoints, tokens)
        assert len(result["similarity_curve"]) == 3  # n-1

    def test_single_checkpoint(self):
        result = detect_phase_transitions([_make_model()], jnp.array([0, 1]))
        assert len(result["similarity_curve"]) == 0


class TestTrackCircuitFormation:
    def test_returns_dict(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        hooks = [("blocks.0.attn.hook_result", lambda x, n: jnp.zeros_like(x))]
        result = track_circuit_formation(checkpoints, tokens, hooks, _metric)
        assert "clean_metrics" in result
        assert "circuit_effects" in result
        assert "formation_checkpoint" in result
        assert "formation_curve" in result

    def test_effect_length(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        hooks = [("blocks.0.attn.hook_result", lambda x, n: jnp.zeros_like(x))]
        result = track_circuit_formation(checkpoints, tokens, hooks, _metric)
        assert len(result["circuit_effects"]) == 3


class TestMeasureRepresentationCrystallization:
    def test_returns_dict(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        result = measure_representation_crystallization(checkpoints, tokens)
        assert "effective_ranks" in result
        assert "condition_numbers" in result
        assert "crystallization_index" in result

    def test_ranks_length(self):
        checkpoints = [_make_model(s) for s in range(4)]
        tokens = jnp.array([0, 1, 2, 3])
        result = measure_representation_crystallization(checkpoints, tokens)
        assert len(result["effective_ranks"]) == 4

    def test_ranks_positive(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        result = measure_representation_crystallization(checkpoints, tokens)
        assert np.all(result["effective_ranks"] > 0)


class TestCompareLearningOrder:
    def test_returns_dict(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        specs = {
            "attn0": [("blocks.0.attn.hook_result", lambda x, n: jnp.zeros_like(x))],
            "attn1": [("blocks.1.attn.hook_result", lambda x, n: jnp.zeros_like(x))],
        }
        result = compare_learning_order(checkpoints, tokens, specs, _metric)
        assert "formation_order" in result
        assert "per_circuit_curves" in result
        assert "earliest_circuit" in result

    def test_all_circuits_tracked(self):
        checkpoints = [_make_model(s) for s in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        specs = {
            "attn0": [("blocks.0.attn.hook_result", lambda x, n: jnp.zeros_like(x))],
            "attn1": [("blocks.1.attn.hook_result", lambda x, n: jnp.zeros_like(x))],
        }
        result = compare_learning_order(checkpoints, tokens, specs, _metric)
        assert len(result["formation_order"]) == 2


class TestGrokkingDetector:
    def test_returns_dict(self):
        checkpoints = [_make_model(s) for s in range(3)]
        train_tokens = jnp.array([0, 1, 2, 3])
        test_tokens = jnp.array([4, 5, 6, 7])
        result = grokking_detector(checkpoints, train_tokens, test_tokens, _metric)
        assert "train_metrics" in result
        assert "test_metrics" in result
        assert "generalization_gap" in result
        assert "grokking_detected" in result
        assert "weight_norms" in result

    def test_metrics_length(self):
        checkpoints = [_make_model(s) for s in range(4)]
        train_tokens = jnp.array([0, 1, 2, 3])
        test_tokens = jnp.array([4, 5, 6, 7])
        result = grokking_detector(checkpoints, train_tokens, test_tokens, _metric)
        assert len(result["train_metrics"]) == 4
        assert len(result["weight_norms"]) == 4

    def test_weight_norms_positive(self):
        checkpoints = [_make_model(s) for s in range(3)]
        train_tokens = jnp.array([0, 1, 2, 3])
        test_tokens = jnp.array([4, 5, 6, 7])
        result = grokking_detector(checkpoints, train_tokens, test_tokens, _metric)
        assert np.all(result["weight_norms"] > 0)
