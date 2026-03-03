"""Tests for logit dynamics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.logit_dynamics import (
    logit_flip_analysis,
    prediction_stability_across_layers,
    commitment_timing,
    logit_contribution_by_component,
    top_k_trajectory,
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


class TestLogitFlipAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_flip_analysis(model, tokens)
        assert "layer_predictions" in result
        assert "flip_layers" in result
        assert "n_flips" in result
        assert "final_prediction" in result
        assert "first_correct_layer" in result

    def test_predictions_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_flip_analysis(model, tokens)
        assert len(result["layer_predictions"]) == 2

    def test_n_flips_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_flip_analysis(model, tokens)
        assert result["n_flips"] >= 0


class TestPredictionStabilityAcrossLayers:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_stability_across_layers(model, tokens)
        assert "layer_top_k" in result
        assert "overlap_with_final" in result
        assert "stability_scores" in result
        assert "mean_stability" in result
        assert "final_top_k" in result

    def test_overlap_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_stability_across_layers(model, tokens)
        assert np.all(result["overlap_with_final"] >= 0)
        assert np.all(result["overlap_with_final"] <= 1.0)

    def test_final_top_k_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_stability_across_layers(model, tokens, top_k=3)
        assert len(result["final_top_k"]) == 3


class TestCommitmentTiming:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = commitment_timing(model, tokens)
        assert "layer_confidence" in result
        assert "commitment_layer" in result
        assert "confidence_growth_rate" in result
        assert "max_confidence_jump" in result
        assert "final_confidence" in result

    def test_confidence_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = commitment_timing(model, tokens)
        assert np.all(result["layer_confidence"] >= 0)
        assert np.all(result["layer_confidence"] <= 1.0)

    def test_final_confidence_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = commitment_timing(model, tokens)
        assert result["final_confidence"] > 0


class TestLogitContributionByComponent:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_contribution_by_component(model, tokens, target_token=5)
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "embedding_contribution" in result
        assert "total_logit" in result
        assert "dominant_component" in result

    def test_contributions_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_contribution_by_component(model, tokens, target_token=5)
        assert len(result["attn_contributions"]) == 2
        assert len(result["mlp_contributions"]) == 2

    def test_dominant_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_contribution_by_component(model, tokens, target_token=5)
        assert result["dominant_component"] in ("attention", "mlp")


class TestTopKTrajectory:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_k_trajectory(model, tokens)
        assert "tracked_tokens" in result
        assert "probability_trajectories" in result
        assert "convergence_layer" in result
        assert "final_probabilities" in result

    def test_tracked_tokens_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_k_trajectory(model, tokens, top_k=3)
        assert len(result["tracked_tokens"]) == 3
        assert len(result["final_probabilities"]) == 3

    def test_trajectories_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_k_trajectory(model, tokens, top_k=3)
        for t in result["tracked_tokens"]:
            assert len(result["probability_trajectories"][t]) == 2
