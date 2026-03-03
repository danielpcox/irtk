"""Tests for weight structure analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.weight_structure import (
    weight_spectral_analysis,
    parameter_utilization,
    head_weight_similarity,
    embedding_weight_relationship,
    layer_weight_norm_profile,
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


class TestWeightSpectralAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        result = weight_spectral_analysis(model, top_k=3)
        assert "attn_spectral" in result
        assert "mlp_spectral" in result
        assert "effective_ranks" in result
        assert "condition_numbers" in result

    def test_has_entries(self):
        model = _make_model()
        result = weight_spectral_analysis(model, top_k=3)
        # Should have entries for 2 layers * 4 attn matrices + 2 layers * 2 mlp matrices
        assert len(result["effective_ranks"]) == 12

    def test_ranks_positive(self):
        model = _make_model()
        result = weight_spectral_analysis(model, top_k=3)
        for k, v in result["effective_ranks"].items():
            assert v >= 0


class TestParameterUtilization:
    def test_returns_dict(self):
        model = _make_model()
        result = parameter_utilization(model)
        assert "total_params" in result
        assert "near_zero_fraction" in result
        assert "weight_norm_by_layer" in result
        assert "param_count_by_type" in result
        assert "weight_magnitude_stats" in result

    def test_total_positive(self):
        model = _make_model()
        result = parameter_utilization(model)
        assert result["total_params"] > 0

    def test_fraction_bounded(self):
        model = _make_model()
        result = parameter_utilization(model)
        assert 0.0 <= result["near_zero_fraction"] <= 1.0

    def test_norms_by_layer(self):
        model = _make_model()
        result = parameter_utilization(model)
        assert len(result["weight_norm_by_layer"]) == 2
        assert np.all(result["weight_norm_by_layer"] > 0)


class TestHeadWeightSimilarity:
    def test_returns_dict(self):
        model = _make_model()
        result = head_weight_similarity(model)
        assert "within_layer_similarity" in result
        assert "across_layer_similarity" in result
        assert "most_similar_heads" in result
        assert "most_different_heads" in result

    def test_within_length(self):
        model = _make_model()
        result = head_weight_similarity(model)
        assert len(result["within_layer_similarity"]) == 2


class TestEmbeddingWeightRelationship:
    def test_returns_dict(self):
        model = _make_model()
        result = embedding_weight_relationship(model)
        assert "weight_tying_score" in result
        assert "frobenius_distance" in result
        assert "rank_correlation" in result
        assert "embed_rank" in result
        assert "unembed_rank" in result

    def test_ranks_positive(self):
        model = _make_model()
        result = embedding_weight_relationship(model)
        assert result["embed_rank"] > 0
        assert result["unembed_rank"] > 0


class TestLayerWeightNormProfile:
    def test_returns_dict(self):
        model = _make_model()
        result = layer_weight_norm_profile(model)
        assert "attn_q_norms" in result
        assert "attn_k_norms" in result
        assert "attn_v_norms" in result
        assert "attn_o_norms" in result
        assert "mlp_in_norms" in result
        assert "mlp_out_norms" in result
        assert "total_per_layer" in result

    def test_shapes(self):
        model = _make_model()
        result = layer_weight_norm_profile(model)
        assert len(result["attn_q_norms"]) == 2
        assert len(result["total_per_layer"]) == 2

    def test_norms_positive(self):
        model = _make_model()
        result = layer_weight_norm_profile(model)
        assert np.all(result["attn_q_norms"] > 0)
        assert np.all(result["total_per_layer"] > 0)
