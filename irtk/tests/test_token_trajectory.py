"""Tests for token_trajectory module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_trajectory import (
    token_representation_trajectory,
    trajectory_velocity,
    trajectory_acceleration,
    token_convergence,
    trajectory_curvature,
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


class TestTokenRepresentationTrajectory:
    def test_basic(self, model, tokens):
        result = token_representation_trajectory(model, tokens)
        assert "trajectories" in result
        assert "norms" in result
        assert "positions_tracked" in result

    def test_shapes(self, model, tokens):
        result = token_representation_trajectory(model, tokens)
        nl = model.cfg.n_layers
        d = model.cfg.d_model
        seq_len = len(tokens)
        assert result["trajectories"].shape == (seq_len, nl + 1, d)
        assert result["norms"].shape == (seq_len, nl + 1)

    def test_subset_positions(self, model, tokens):
        result = token_representation_trajectory(model, tokens, positions=[0, 2])
        assert result["trajectories"].shape[0] == 2
        assert result["positions_tracked"] == [0, 2]


class TestTrajectoryVelocity:
    def test_basic(self, model, tokens):
        result = trajectory_velocity(model, tokens)
        assert "velocities" in result
        assert "directions" in result
        assert "mean_velocity" in result
        assert "fastest_layer" in result

    def test_shapes(self, model, tokens):
        result = trajectory_velocity(model, tokens)
        nl = model.cfg.n_layers
        d = model.cfg.d_model
        seq_len = len(tokens)
        assert result["velocities"].shape == (seq_len, nl)
        assert result["directions"].shape == (seq_len, nl, d)
        assert result["mean_velocity"].shape == (nl,)

    def test_velocities_nonneg(self, model, tokens):
        result = trajectory_velocity(model, tokens)
        assert np.all(result["velocities"] >= 0)


class TestTrajectoryAcceleration:
    def test_basic(self, model, tokens):
        result = trajectory_acceleration(model, tokens)
        assert "accelerations" in result
        assert "mean_acceleration" in result
        assert "is_decelerating" in result

    def test_shapes(self, model, tokens):
        result = trajectory_acceleration(model, tokens)
        nl = model.cfg.n_layers
        seq_len = len(tokens)
        assert result["accelerations"].shape == (seq_len, nl - 1)
        assert result["mean_acceleration"].shape == (nl - 1,)


class TestTokenConvergence:
    def test_basic(self, model, tokens):
        result = token_convergence(model, tokens, pos_a=0, pos_b=-1)
        assert "distances" in result
        assert "cosine_similarities" in result
        assert "is_converging" in result
        assert "convergence_rate" in result

    def test_shapes(self, model, tokens):
        result = token_convergence(model, tokens)
        nl = model.cfg.n_layers
        assert result["distances"].shape == (nl + 1,)
        assert result["cosine_similarities"].shape == (nl + 1,)

    def test_distances_nonneg(self, model, tokens):
        result = token_convergence(model, tokens)
        assert np.all(result["distances"] >= 0)

    def test_cosine_range(self, model, tokens):
        result = token_convergence(model, tokens)
        assert np.all(result["cosine_similarities"] >= -1.01)
        assert np.all(result["cosine_similarities"] <= 1.01)


class TestTrajectoryCurvature:
    def test_basic(self, model, tokens):
        result = trajectory_curvature(model, tokens)
        assert "curvatures" in result
        assert "mean_curvature" in result
        assert "straightest_position" in result
        assert "most_curved_position" in result

    def test_shapes(self, model, tokens):
        result = trajectory_curvature(model, tokens)
        nl = model.cfg.n_layers
        seq_len = len(tokens)
        assert result["curvatures"].shape == (seq_len, nl - 1)

    def test_curvatures_nonneg(self, model, tokens):
        result = trajectory_curvature(model, tokens)
        assert np.all(result["curvatures"] >= 0)
