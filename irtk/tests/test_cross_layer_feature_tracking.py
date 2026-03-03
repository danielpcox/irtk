"""Tests for cross_layer_feature_tracking module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.cross_layer_feature_tracking import (
    feature_persistence,
    transformation_analysis,
    creation_destruction_events,
    feature_lineage,
    representation_drift,
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


class TestFeaturePersistence:
    def test_basic(self, model, tokens):
        result = feature_persistence(model, tokens)
        assert "persistence_scores" in result
        assert "projection_magnitudes" in result
        assert "peak_layer" in result
        assert "decay_rate" in result

    def test_persistence_shape(self, model, tokens):
        result = feature_persistence(model, tokens)
        assert result["persistence_scores"].shape == (model.cfg.n_layers,)


class TestTransformationAnalysis:
    def test_basic(self, model, tokens):
        result = transformation_analysis(model, tokens)
        assert "inter_layer_cosines" in result
        assert "transformation_norms" in result
        assert "rotation_angles" in result
        assert "stretch_factors" in result

    def test_shapes(self, model, tokens):
        result = transformation_analysis(model, tokens)
        nl = model.cfg.n_layers - 1
        assert result["inter_layer_cosines"].shape == (nl,)
        assert result["rotation_angles"].shape == (nl,)


class TestCreationDestructionEvents:
    def test_basic(self, model, tokens):
        result = creation_destruction_events(model, tokens)
        assert "creation_layers" in result
        assert "destruction_layers" in result
        assert "dimension_utilization" in result

    def test_utilization_shape(self, model, tokens):
        result = creation_destruction_events(model, tokens)
        assert result["dimension_utilization"].shape == (model.cfg.n_layers,)


class TestFeatureLineage:
    def test_basic(self, model, tokens):
        result = feature_lineage(model, tokens, source_layer=0, n_directions=2)
        assert "direction_persistence" in result
        assert "most_persistent_direction" in result
        assert "mean_persistence" in result

    def test_shape(self, model, tokens):
        result = feature_lineage(model, tokens, source_layer=0, n_directions=2)
        assert result["direction_persistence"].shape == (2, model.cfg.n_layers)


class TestRepresentationDrift:
    def test_basic(self, model):
        tokens_list = [jnp.array([0, 5, 10, 15, 20]), jnp.array([1, 6, 11, 16, 21])]
        result = representation_drift(model, tokens_list)
        assert "mean_drift_per_layer" in result
        assert "convergence_rate" in result
        assert "total_drift" in result

    def test_shape(self, model):
        tokens_list = [jnp.array([0, 5, 10, 15, 20]), jnp.array([1, 6, 11, 16, 21])]
        result = representation_drift(model, tokens_list)
        assert result["mean_drift_per_layer"].shape == (model.cfg.n_layers - 1,)
