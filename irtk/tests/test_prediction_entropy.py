"""Tests for prediction entropy and information budget analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.prediction_entropy import (
    layer_prediction_entropy,
    prediction_commit_depth,
    per_token_surprisal,
    entropy_reduction_by_layer,
    compare_entropy_profiles,
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


# ─── Layer Prediction Entropy ──────────────────────────────────────────────


class TestLayerPredictionEntropy:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_prediction_entropy(model, tokens)
        # n_layers+1 components x seq_len
        assert result.shape == (3, 4)  # 2 layers + embed = 3

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_prediction_entropy(model, tokens)
        assert np.all(result >= -1e-6)

    def test_different_inputs_differ(self):
        model = _make_model()
        r1 = layer_prediction_entropy(model, jnp.array([0, 1, 2, 3]))
        r2 = layer_prediction_entropy(model, jnp.array([10, 20, 30, 40]))
        assert not np.allclose(r1, r2, atol=1e-5)

    def test_no_ln(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_prediction_entropy(model, tokens, apply_ln=False)
        assert result.shape == (3, 4)


# ─── Prediction Commit Depth ──────────────────────────────────────────────


class TestPredictionCommitDepth:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_commit_depth(model, tokens, target_token=5, pos=-1)
        assert "commit_layer" in result
        assert "probability_trajectory" in result
        assert "top1_trajectory" in result
        assert "final_probability" in result

    def test_trajectory_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_commit_depth(model, tokens, target_token=5)
        assert len(result["probability_trajectory"]) == 3  # n_layers+1
        assert len(result["top1_trajectory"]) == 3

    def test_prob_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_commit_depth(model, tokens, target_token=0)
        for p in result["probability_trajectory"]:
            assert 0 <= p <= 1.0

    def test_high_threshold_may_not_commit(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_commit_depth(model, tokens, target_token=49, threshold=0.99)
        # With random weights, unlikely to commit at 99% threshold
        # commit_layer can be None or an int
        assert result["commit_layer"] is None or isinstance(result["commit_layer"], int)

    def test_final_probability_is_float(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_commit_depth(model, tokens, target_token=0)
        assert isinstance(result["final_probability"], float)


# ─── Per-Token Surprisal ──────────────────────────────────────────────────


class TestPerTokenSurprisal:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_token_surprisal(model, tokens)
        assert "surprisals" in result
        assert "mean_surprisal" in result
        assert "most_surprising_pos" in result

    def test_surprisal_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_token_surprisal(model, tokens)
        assert len(result["surprisals"]) == 3  # seq_len - 1

    def test_surprisal_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_token_surprisal(model, tokens)
        assert np.all(result["surprisals"] >= -1e-6)

    def test_mean_is_mean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_token_surprisal(model, tokens)
        assert abs(result["mean_surprisal"] - float(np.mean(result["surprisals"]))) < 1e-5

    def test_positions_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_token_surprisal(model, tokens)
        assert 0 <= result["most_surprising_pos"] < 3
        assert 0 <= result["least_surprising_pos"] < 3


# ─── Entropy Reduction by Layer ───────────────────────────────────────────


class TestEntropyReductionByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = entropy_reduction_by_layer(model, tokens, pos=-1)
        assert "entropy_per_layer" in result
        assert "delta_entropy" in result
        assert "biggest_reducer" in result
        assert "total_reduction" in result

    def test_entropy_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = entropy_reduction_by_layer(model, tokens)
        assert len(result["entropy_per_layer"]) == 3  # n_layers+1
        assert len(result["delta_entropy"]) == 2  # n_layers

    def test_delta_sums_to_total(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = entropy_reduction_by_layer(model, tokens)
        total = float(np.sum(result["delta_entropy"]))
        expected = result["entropy_per_layer"][0] - result["entropy_per_layer"][-1]
        assert abs(total - expected) < 1e-5

    def test_biggest_reducer_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = entropy_reduction_by_layer(model, tokens)
        assert 0 <= result["biggest_reducer"] < 2  # n_layers


# ─── Compare Entropy Profiles ─────────────────────────────────────────────


class TestCompareEntropyProfiles:
    def test_returns_dict(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([10, 20, 30, 40])
        result = compare_entropy_profiles(model, a, b, pos=-1)
        assert "entropy_a" in result
        assert "entropy_b" in result
        assert "absolute_diff" in result
        assert "max_diff_layer" in result

    def test_profile_length(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([10, 20, 30, 40])
        result = compare_entropy_profiles(model, a, b)
        assert len(result["entropy_a"]) == 3
        assert len(result["entropy_b"]) == 3
        assert len(result["absolute_diff"]) == 3

    def test_same_input_zero_diff(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compare_entropy_profiles(model, tokens, tokens)
        assert np.allclose(result["absolute_diff"], 0, atol=1e-5)

    def test_diff_nonneg(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([10, 20, 30, 40])
        result = compare_entropy_profiles(model, a, b)
        assert np.all(result["absolute_diff"] >= -1e-6)

    def test_max_diff_layer_valid(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([10, 20, 30, 40])
        result = compare_entropy_profiles(model, a, b)
        assert 0 <= result["max_diff_layer"] < 3
