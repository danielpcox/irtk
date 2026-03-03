"""Tests for activation_diffing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_diffing import (
    paired_activation_comparison,
    change_localization,
    divergence_mapping,
    causal_change_attribution,
    minimal_change_identification,
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
def tokens_a():
    return jnp.array([0, 5, 10, 15, 20])


@pytest.fixture
def tokens_b():
    return jnp.array([25, 30, 35, 40, 45])


def metric_fn(logits, tokens):
    return jnp.mean(logits[-1])


class TestPairedActivationComparison:
    def test_basic(self, model, tokens_a, tokens_b):
        result = paired_activation_comparison(model, tokens_a, tokens_b)
        assert "per_layer" in result
        assert len(result["per_layer"]) == model.cfg.n_layers

    def test_per_layer_keys(self, model, tokens_a, tokens_b):
        result = paired_activation_comparison(model, tokens_a, tokens_b)
        for layer_info in result["per_layer"]:
            assert "l2_distance" in layer_info
            assert "cosine_similarity" in layer_info


class TestChangeLocalization:
    def test_basic(self, model, tokens_a, tokens_b):
        result = change_localization(model, tokens_a, tokens_b)
        assert "component_changes" in result
        assert "most_changed_component" in result

    def test_has_components(self, model, tokens_a, tokens_b):
        result = change_localization(model, tokens_a, tokens_b)
        assert len(result["component_changes"]) > 0


class TestDivergenceMapping:
    def test_basic(self, model, tokens_a, tokens_b):
        result = divergence_mapping(model, tokens_a, tokens_b)
        assert "divergence_matrix" in result
        assert "peak_divergence" in result
        assert "mean_per_layer" in result

    def test_matrix_shape(self, model, tokens_a, tokens_b):
        result = divergence_mapping(model, tokens_a, tokens_b)
        assert result["divergence_matrix"].shape == (model.cfg.n_layers, len(tokens_a))


class TestCausalChangeAttribution:
    def test_basic(self, model, tokens_a, tokens_b):
        result = causal_change_attribution(model, tokens_a, tokens_b, metric_fn)
        assert "component_attributions" in result
        assert "total_change" in result
        assert "top_components" in result

    def test_attributions_populated(self, model, tokens_a, tokens_b):
        result = causal_change_attribution(model, tokens_a, tokens_b, metric_fn)
        assert len(result["component_attributions"]) > 0


class TestMinimalChangeIdentification:
    def test_basic(self, model, tokens_a, tokens_b):
        result = minimal_change_identification(model, tokens_a, tokens_b, top_k=3)
        assert "top_dimensions" in result
        assert "cumulative_explanation" in result
        assert "total_diff_norm" in result

    def test_top_k(self, model, tokens_a, tokens_b):
        result = minimal_change_identification(model, tokens_a, tokens_b, top_k=3)
        assert len(result["top_dimensions"]) == 3

    def test_cumulative_bounded(self, model, tokens_a, tokens_b):
        result = minimal_change_identification(model, tokens_a, tokens_b, top_k=5)
        if result["cumulative_explanation"]:
            assert result["cumulative_explanation"][-1] <= 1.01
