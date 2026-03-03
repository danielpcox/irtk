"""Tests for token_prediction_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_prediction_analysis import (
    per_token_confidence,
    surprisal_profile,
    token_difficulty_profile,
    prediction_agreement_by_layer,
    rank_trajectory,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


class TestPerTokenConfidence:
    def test_output_keys(self, model, tokens):
        r = per_token_confidence(model, tokens)
        assert "top1_probs" in r
        assert "top1_tokens" in r
        assert "entropies" in r
        assert "mean_confidence" in r
        assert "least_confident_position" in r
        assert "most_confident_position" in r

    def test_shapes(self, model, tokens):
        r = per_token_confidence(model, tokens)
        seq_len = len(tokens)
        assert r["top1_probs"].shape == (seq_len,)
        assert r["top1_tokens"].shape == (seq_len,)
        assert r["entropies"].shape == (seq_len,)

    def test_probs_valid(self, model, tokens):
        r = per_token_confidence(model, tokens)
        assert np.all(r["top1_probs"] > 0)
        assert np.all(r["top1_probs"] <= 1.0 + 1e-5)

    def test_entropies_nonneg(self, model, tokens):
        r = per_token_confidence(model, tokens)
        assert np.all(r["entropies"] >= -1e-5)

    def test_positions_valid(self, model, tokens):
        r = per_token_confidence(model, tokens)
        seq_len = len(tokens)
        assert 0 <= r["least_confident_position"] < seq_len
        assert 0 <= r["most_confident_position"] < seq_len


class TestSurprisalProfile:
    def test_output_keys(self, model, tokens):
        r = surprisal_profile(model, tokens)
        assert "surprisals" in r
        assert "mean_surprisal" in r
        assert "correct_predictions" in r
        assert "accuracy" in r

    def test_shapes(self, model, tokens):
        r = surprisal_profile(model, tokens)
        assert r["surprisals"].shape == (len(tokens) - 1,)
        assert r["correct_predictions"].shape == (len(tokens) - 1,)

    def test_surprisals_nonneg(self, model, tokens):
        r = surprisal_profile(model, tokens)
        assert np.all(r["surprisals"] >= -1e-5)

    def test_accuracy_bounded(self, model, tokens):
        r = surprisal_profile(model, tokens)
        assert 0.0 <= r["accuracy"] <= 1.0


class TestTokenDifficultyProfile:
    def test_output_keys(self, model, tokens):
        r = token_difficulty_profile(model, tokens, n_samples=2)
        assert "position_entropy_mean" in r
        assert "position_entropy_std" in r
        assert "hardest_position" in r
        assert "easiest_position" in r
        assert "relative_difficulty" in r

    def test_shapes(self, model, tokens):
        r = token_difficulty_profile(model, tokens, n_samples=2)
        seq_len = len(tokens)
        assert r["position_entropy_mean"].shape == (seq_len,)
        assert r["position_entropy_std"].shape == (seq_len,)
        assert r["relative_difficulty"].shape == (seq_len,)

    def test_relative_difficulty_bounded(self, model, tokens):
        r = token_difficulty_profile(model, tokens, n_samples=2)
        assert np.all(r["relative_difficulty"] >= -1e-5)
        assert np.all(r["relative_difficulty"] <= 1.0 + 1e-5)


class TestPredictionAgreementByLayer:
    def test_output_keys(self, model, tokens):
        r = prediction_agreement_by_layer(model, tokens, top_k=3)
        assert "layer_top_k" in r
        assert "agreement_with_final" in r
        assert "first_agreement_layer" in r
        assert "consensus_fraction" in r

    def test_shapes(self, model, tokens):
        r = prediction_agreement_by_layer(model, tokens, top_k=3)
        n_layers = model.cfg.n_layers
        assert r["layer_top_k"].shape == (n_layers + 1, 3)
        assert r["agreement_with_final"].shape == (n_layers,)

    def test_agreement_bounded(self, model, tokens):
        r = prediction_agreement_by_layer(model, tokens, top_k=3)
        assert np.all(r["agreement_with_final"] >= 0.0)
        assert np.all(r["agreement_with_final"] <= 1.0 + 1e-5)

    def test_consensus_bounded(self, model, tokens):
        r = prediction_agreement_by_layer(model, tokens, top_k=3)
        assert 0.0 <= r["consensus_fraction"] <= 1.0 + 1e-5


class TestRankTrajectory:
    def test_output_keys(self, model, tokens):
        r = rank_trajectory(model, tokens, target_tokens=[0, 5, 10])
        assert "ranks" in r
        assert "logits" in r
        assert "final_ranks" in r

    def test_shapes(self, model, tokens):
        r = rank_trajectory(model, tokens, target_tokens=[0, 5])
        n_layers = model.cfg.n_layers
        assert r["ranks"][0].shape == (n_layers + 1,)
        assert r["logits"][0].shape == (n_layers + 1,)

    def test_ranks_nonneg(self, model, tokens):
        r = rank_trajectory(model, tokens, target_tokens=[0, 5, 10])
        for t in [0, 5, 10]:
            assert np.all(r["ranks"][t] >= 0)

    def test_final_ranks_present(self, model, tokens):
        targets = [0, 5, 10]
        r = rank_trajectory(model, tokens, target_tokens=targets)
        for t in targets:
            assert t in r["final_ranks"]
