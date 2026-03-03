"""Tests for residual stream analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.residual_stream import (
    cosine_similarity_to_unembed,
    residual_norm_by_layer,
    residual_direction_analysis,
    token_prediction_trajectory,
    prediction_rank_trajectory,
    layer_contribution_to_logit,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestCosineSimilarityToUnembed:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        sim = cosine_similarity_to_unembed(model, cache, token=5)
        # embed + n_layers entries
        assert sim.shape == (3,)  # 1 embed + 2 layers

    def test_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        sim = cosine_similarity_to_unembed(model, cache, token=5)
        assert np.all(sim >= -1.0 - 1e-5)
        assert np.all(sim <= 1.0 + 1e-5)


class TestResidualNormByLayer:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        norms = residual_norm_by_layer(cache)
        assert norms.shape == (3,)  # embed + 2 layers

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        norms = residual_norm_by_layer(cache)
        assert np.all(norms >= 0)

    def test_specific_position(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        norms = residual_norm_by_layer(cache, pos=0)
        assert norms.shape == (3,)


class TestResidualDirectionAnalysis:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        direction = jnp.ones(model.cfg.d_model)
        result = residual_direction_analysis(model, cache, direction)
        assert "components" in result
        assert "labels" in result
        assert "cumulative" in result

    def test_components_match_labels(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        direction = jnp.ones(model.cfg.d_model)
        result = residual_direction_analysis(model, cache, direction)
        assert len(result["components"]) == len(result["labels"])
        assert len(result["cumulative"]) == len(result["labels"])

    def test_cumulative_is_cumsum(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        direction = jnp.ones(model.cfg.d_model)
        result = residual_direction_analysis(model, cache, direction)
        np.testing.assert_allclose(
            result["cumulative"],
            np.cumsum(result["components"]),
            atol=1e-5,
        )


class TestTokenPredictionTrajectory:
    def test_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        traj = token_prediction_trajectory(model, tokens, k=3)
        assert len(traj) == 3  # embed + 2 layers

    def test_k_predictions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        traj = token_prediction_trajectory(model, tokens, k=5)
        for layer_preds in traj:
            assert len(layer_preds) == 5

    def test_probabilities_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        traj = token_prediction_trajectory(model, tokens, k=3)
        for layer_preds in traj:
            for token_id, prob in layer_preds:
                assert 0 <= prob <= 1
                assert 0 <= token_id < model.cfg.d_vocab


class TestPredictionRankTrajectory:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        ranks = prediction_rank_trajectory(model, tokens, target_token=5)
        assert ranks.shape == (3,)  # embed + 2 layers

    def test_valid_ranks(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        ranks = prediction_rank_trajectory(model, tokens, target_token=5)
        assert np.all(ranks >= 0)
        assert np.all(ranks < model.cfg.d_vocab)


class TestLayerContributionToLogit:
    def test_returns_components(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        result = layer_contribution_to_logit(model, cache, token=5)
        assert "embed" in result
        assert isinstance(result["embed"], float)
