"""Tests for layer composition analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.layer_composition import (
    layer_residual_contribution,
    layer_output_similarity,
    layer_redundancy_analysis,
    critical_layer_identification,
    layer_specialization_profile,
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


def _metric(logits):
    return float(logits[-1, 0])


class TestLayerResidualContribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_residual_contribution(model, tokens)
        assert "layer_contributions" in result
        assert "layer_logit_norms" in result
        assert "embedding_contribution" in result
        assert "dominant_layer" in result
        assert "contribution_fractions" in result

    def test_contributions_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_residual_contribution(model, tokens)
        assert np.all(result["layer_contributions"] >= 0)

    def test_fractions_sum_near_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_residual_contribution(model, tokens)
        assert np.sum(result["contribution_fractions"]) <= 1.0 + 0.01


class TestLayerOutputSimilarity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_output_similarity(model, tokens)
        assert "consecutive_similarity" in result
        assert "similarity_to_final" in result
        assert "mean_consecutive_similarity" in result
        assert "most_different_layer" in result

    def test_similarity_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_output_similarity(model, tokens)
        assert np.all(result["consecutive_similarity"] >= -1.0 - 1e-5)
        assert np.all(result["consecutive_similarity"] <= 1.0 + 1e-5)


class TestLayerRedundancyAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_redundancy_analysis(model, tokens, _metric)
        assert "ablation_effects" in result
        assert "most_redundant_layer" in result
        assert "most_critical_layer" in result
        assert "redundancy_scores" in result

    def test_effects_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_redundancy_analysis(model, tokens, _metric)
        assert np.all(result["ablation_effects"] >= 0)


class TestCriticalLayerIdentification:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_layer_identification(model, tokens, _metric)
        assert "layer_effects" in result
        assert "critical_layers" in result
        assert "dispensable_layers" in result
        assert "n_critical" in result
        assert "baseline_metric" in result

    def test_layers_partition(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = critical_layer_identification(model, tokens, _metric)
        total = len(result["critical_layers"]) + len(result["dispensable_layers"])
        assert total == 2  # n_layers


class TestLayerSpecializationProfile:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_specialization_profile(model, tokens)
        assert "attn_norms" in result
        assert "mlp_norms" in result
        assert "attn_fraction" in result
        assert "mlp_dominant_layers" in result
        assert "attn_dominant_layers" in result

    def test_norms_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_specialization_profile(model, tokens)
        assert np.all(result["attn_norms"] >= 0)
        assert np.all(result["mlp_norms"] >= 0)

    def test_fractions_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layer_specialization_profile(model, tokens)
        assert np.all(result["attn_fraction"] >= 0)
        assert np.all(result["attn_fraction"] <= 1.0)
