"""Tests for embedding_dynamics module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.embedding_dynamics import (
    token_identity_decay,
    semantic_drift_analysis,
    positional_encoding_persistence,
    embedding_subspace_tracking,
    context_mixing_rate,
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


class TestTokenIdentityDecay:
    def test_output_keys(self, model, tokens):
        r = token_identity_decay(model, tokens)
        assert "identity_trajectory" in r
        assert "half_life_layer" in r
        assert "decay_rate" in r
        assert "final_identity" in r

    def test_shapes(self, model, tokens):
        r = token_identity_decay(model, tokens)
        assert r["identity_trajectory"].shape == (model.cfg.n_layers + 1,)

    def test_trajectory_bounded(self, model, tokens):
        r = token_identity_decay(model, tokens)
        assert np.all(r["identity_trajectory"] >= -1.0 - 1e-5)
        assert np.all(r["identity_trajectory"] <= 1.0 + 1e-5)


class TestSemanticDriftAnalysis:
    def test_output_keys(self, model, tokens):
        r = semantic_drift_analysis(model, tokens)
        assert "drift_angles" in r
        assert "cumulative_drift" in r
        assert "fastest_drift_layer" in r
        assert "total_drift" in r

    def test_shapes(self, model, tokens):
        r = semantic_drift_analysis(model, tokens)
        assert r["drift_angles"].shape == (model.cfg.n_layers,)
        assert r["cumulative_drift"].shape == (model.cfg.n_layers,)

    def test_angles_nonneg(self, model, tokens):
        r = semantic_drift_analysis(model, tokens)
        assert np.all(r["drift_angles"] >= -1e-8)

    def test_cumulative_monotonic(self, model, tokens):
        r = semantic_drift_analysis(model, tokens)
        assert np.all(np.diff(r["cumulative_drift"]) >= -1e-8)


class TestPositionalEncodingPersistence:
    def test_output_keys(self, model, tokens):
        r = positional_encoding_persistence(model, tokens)
        assert "position_discriminability" in r
        assert "mean_inter_position_distance" in r
        assert "position_order_preserved" in r
        assert "persistence_score" in r

    def test_shapes(self, model, tokens):
        r = positional_encoding_persistence(model, tokens)
        assert r["position_discriminability"].shape == (model.cfg.n_layers + 1,)
        assert r["mean_inter_position_distance"].shape == (model.cfg.n_layers + 1,)


class TestEmbeddingSubspaceTracking:
    def test_output_keys(self, model, tokens):
        r = embedding_subspace_tracking(model, tokens, n_components=3)
        assert "embedding_subspace_projection" in r
        assert "orthogonal_growth" in r
        assert "subspace_exit_layer" in r
        assert "final_subspace_fraction" in r

    def test_shapes(self, model, tokens):
        r = embedding_subspace_tracking(model, tokens, n_components=3)
        assert r["embedding_subspace_projection"].shape == (model.cfg.n_layers + 1,)

    def test_projection_bounded(self, model, tokens):
        r = embedding_subspace_tracking(model, tokens, n_components=3)
        assert np.all(r["embedding_subspace_projection"] >= -1e-5)
        assert np.all(r["embedding_subspace_projection"] <= 1.0 + 1e-5)


class TestContextMixingRate:
    def test_output_keys(self, model, tokens):
        r = context_mixing_rate(model, tokens, target_pos=-1)
        assert "self_similarity" in r
        assert "mixing_rate" in r
        assert "context_dependence" in r
        assert "fastest_mixing_layer" in r

    def test_shapes(self, model, tokens):
        r = context_mixing_rate(model, tokens)
        assert r["self_similarity"].shape == (model.cfg.n_layers + 1,)
        assert r["mixing_rate"].shape == (model.cfg.n_layers,)

    def test_mixing_nonneg(self, model, tokens):
        r = context_mixing_rate(model, tokens)
        assert np.all(r["mixing_rate"] >= -1e-5)
