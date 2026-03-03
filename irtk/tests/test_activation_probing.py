"""Tests for activation_probing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_probing import (
    multiclass_probe,
    nonlinear_probe,
    concept_localization,
    cross_layer_probe_transfer,
    probe_selectivity,
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
def tokens_list():
    return [
        jnp.array([0, 5, 10, 15, 20]),
        jnp.array([1, 6, 11, 16, 21]),
        jnp.array([2, 7, 12, 17, 22]),
        jnp.array([3, 8, 13, 18, 23]),
    ]


@pytest.fixture
def labels():
    return [0, 1, 0, 1]


class TestMulticlassProbe:
    def test_basic(self, model, tokens_list, labels):
        result = multiclass_probe(model, tokens_list, labels)
        assert "accuracy" in result
        assert "class_accuracies" in result
        assert "most_discriminative_dims" in result
        assert "confusion_matrix" in result

    def test_accuracy_range(self, model, tokens_list, labels):
        result = multiclass_probe(model, tokens_list, labels)
        assert 0 <= result["accuracy"] <= 1

    def test_confusion_shape(self, model, tokens_list, labels):
        result = multiclass_probe(model, tokens_list, labels)
        n_classes = 2
        assert result["confusion_matrix"].shape == (n_classes, n_classes)


class TestNonlinearProbe:
    def test_basic(self, model, tokens_list, labels):
        result = nonlinear_probe(model, tokens_list, labels)
        assert "accuracy" in result
        assert "linear_accuracy" in result
        assert "nonlinearity_gain" in result

    def test_accuracy_range(self, model, tokens_list, labels):
        result = nonlinear_probe(model, tokens_list, labels)
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["linear_accuracy"] <= 1


class TestConceptLocalization:
    def test_basic(self, model, tokens_list, labels):
        result = concept_localization(model, tokens_list, labels)
        assert "layer_accuracies" in result
        assert "emergence_layer" in result
        assert "peak_layer" in result

    def test_shapes(self, model, tokens_list, labels):
        result = concept_localization(model, tokens_list, labels)
        nl = model.cfg.n_layers
        assert result["layer_accuracies"].shape == (nl,)

    def test_valid_layers(self, model, tokens_list, labels):
        result = concept_localization(model, tokens_list, labels)
        nl = model.cfg.n_layers
        assert 0 <= result["emergence_layer"] < nl
        assert 0 <= result["peak_layer"] < nl


class TestCrossLayerProbeTransfer:
    def test_basic(self, model, tokens_list, labels):
        result = cross_layer_probe_transfer(model, tokens_list, labels, train_layer=0)
        assert "train_accuracy" in result
        assert "transfer_accuracies" in result
        assert "best_transfer_layer" in result
        assert "representation_similarity" in result

    def test_shapes(self, model, tokens_list, labels):
        result = cross_layer_probe_transfer(model, tokens_list, labels, train_layer=0)
        nl = model.cfg.n_layers
        assert result["transfer_accuracies"].shape == (nl,)
        assert result["representation_similarity"].shape == (nl,)


class TestProbeSelectivity:
    def test_basic(self, model, tokens_list, labels):
        result = probe_selectivity(model, tokens_list, labels)
        assert "selectivity_score" in result
        assert "class_separations" in result
        assert "dimension_usage" in result
        assert "noise_ratio" in result

    def test_selectivity_range(self, model, tokens_list, labels):
        result = probe_selectivity(model, tokens_list, labels)
        assert 0 <= result["selectivity_score"] <= 1

    def test_noise_nonneg(self, model, tokens_list, labels):
        result = probe_selectivity(model, tokens_list, labels)
        assert result["noise_ratio"] >= 0
