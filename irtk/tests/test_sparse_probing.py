"""Tests for sparse_probing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.sparse_probing import (
    sparse_linear_probe,
    sparse_concept_direction,
    feature_selection_probe,
    minimal_probe_set,
    sparse_vs_dense_comparison,
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
    return [jnp.array([0, 5, 10, 15, 20]), jnp.array([1, 6, 11, 16, 21]),
            jnp.array([25, 30, 35, 40, 45]), jnp.array([26, 31, 36, 41, 46])]


@pytest.fixture
def labels():
    return [0, 0, 1, 1]


class TestSparseLinearProbe:
    def test_basic(self, model, tokens_list, labels):
        result = sparse_linear_probe(model, tokens_list, labels)
        assert "accuracy" in result
        assert "sparsity" in result
        assert "active_dimensions" in result

    def test_accuracy_range(self, model, tokens_list, labels):
        result = sparse_linear_probe(model, tokens_list, labels)
        assert 0 <= result["accuracy"] <= 1

    def test_sparsity_range(self, model, tokens_list, labels):
        result = sparse_linear_probe(model, tokens_list, labels)
        assert 0 <= result["sparsity"] <= 1


class TestSparseConceptDirection:
    def test_basic(self, model, tokens_list, labels):
        result = sparse_concept_direction(model, tokens_list, labels, max_dims=3)
        assert "direction" in result
        assert "selected_dims" in result
        assert "accuracy_curve" in result

    def test_dims_count(self, model, tokens_list, labels):
        result = sparse_concept_direction(model, tokens_list, labels, max_dims=3)
        assert len(result["selected_dims"]) == 3


class TestFeatureSelectionProbe:
    def test_basic(self, model, tokens_list, labels):
        result = feature_selection_probe(model, tokens_list, labels, max_features=5)
        assert "selected_features" in result
        assert "accuracy_per_feature" in result
        assert "final_accuracy" in result

    def test_features_bounded(self, model, tokens_list, labels):
        result = feature_selection_probe(model, tokens_list, labels, max_features=3)
        assert len(result["selected_features"]) <= 3


class TestMinimalProbeSet:
    def test_basic(self, model, tokens_list, labels):
        result = minimal_probe_set(model, tokens_list, labels)
        assert "minimal_dims" in result
        assert "n_dims_needed" in result
        assert "full_accuracy" in result

    def test_accuracy_range(self, model, tokens_list, labels):
        result = minimal_probe_set(model, tokens_list, labels)
        assert 0 <= result["full_accuracy"] <= 1


class TestSparseVsDenseComparison:
    def test_basic(self, model, tokens_list, labels):
        result = sparse_vs_dense_comparison(model, tokens_list, labels, n_sparse_dims=3)
        assert "sparse_accuracy" in result
        assert "dense_accuracy" in result
        assert "gap" in result
        assert "efficiency_ratio" in result

    def test_accuracy_range(self, model, tokens_list, labels):
        result = sparse_vs_dense_comparison(model, tokens_list, labels)
        assert 0 <= result["sparse_accuracy"] <= 1
        assert 0 <= result["dense_accuracy"] <= 1
