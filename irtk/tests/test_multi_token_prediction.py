"""Tests for multi-token prediction analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.multi_token_prediction import (
    future_token_probing,
    planning_horizon,
    next_k_token_accuracy,
    representation_lookahead,
    future_information_by_layer,
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


class TestFutureTokenProbing:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_token_probing(model, tokens)
        assert "future_ranks" in result
        assert "future_probs" in result
        assert "mean_rank" in result
        assert "mean_prob" in result
        assert "top_10_fraction" in result

    def test_ranks_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_token_probing(model, tokens, future_offset=1)
        assert len(result["future_ranks"]) == 5  # seq_len - offset

    def test_probs_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_token_probing(model, tokens)
        assert np.all(result["future_probs"] >= 0)
        assert np.all(result["future_probs"] <= 1.0)

    def test_top_10_fraction_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_token_probing(model, tokens)
        assert 0.0 <= result["top_10_fraction"] <= 1.0


class TestPlanningHorizon:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = planning_horizon(model, tokens)
        assert "offset_probs" in result
        assert "offset_ranks" in result
        assert "effective_horizon" in result
        assert "horizon_decay_rate" in result

    def test_offsets_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = planning_horizon(model, tokens, max_lookahead=3)
        assert len(result["offset_probs"]) == 3
        assert len(result["offset_ranks"]) == 3

    def test_horizon_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = planning_horizon(model, tokens)
        assert result["effective_horizon"] >= 0
        assert result["horizon_decay_rate"] >= 0


class TestNextKTokenAccuracy:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = next_k_token_accuracy(model, tokens)
        assert "top_1_accuracy_per_offset" in result
        assert "top_5_accuracy_per_offset" in result
        assert "mean_top_1" in result
        assert "best_offset" in result

    def test_accuracy_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = next_k_token_accuracy(model, tokens, k=2)
        assert np.all(result["top_1_accuracy_per_offset"] >= 0)
        assert np.all(result["top_1_accuracy_per_offset"] <= 1.0)
        assert np.all(result["top_5_accuracy_per_offset"] >= 0)
        assert np.all(result["top_5_accuracy_per_offset"] <= 1.0)


class TestRepresentationLookahead:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = representation_lookahead(model, tokens)
        assert "similarity_to_future" in result
        assert "mean_future_similarity" in result
        assert "lookahead_distance" in result
        assert "max_similarity_offset" in result

    def test_similarity_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = representation_lookahead(model, tokens)
        assert np.all(result["similarity_to_future"] >= -1.0)
        assert np.all(result["similarity_to_future"] <= 1.0)


class TestFutureInformationByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_information_by_layer(model, tokens)
        assert "layer_mean_probs" in result
        assert "layer_mean_ranks" in result
        assert "best_layer" in result
        assert "emergence_layer" in result

    def test_probs_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_information_by_layer(model, tokens)
        assert len(result["layer_mean_probs"]) == 2
        assert len(result["layer_mean_ranks"]) == 2

    def test_probs_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = future_information_by_layer(model, tokens)
        assert np.all(result["layer_mean_probs"] >= 0)
        assert np.all(result["layer_mean_probs"] <= 1.0)
