"""Tests for attention pattern analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.attention_pattern_analysis import (
    attention_entropy_profile,
    positional_attention_bias,
    attention_sparsity_analysis,
    cross_head_attention_similarity,
    attention_head_classification,
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


class TestAttentionEntropyProfile:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_entropy_profile(model, tokens)
        assert "entropy_matrix" in result
        assert "sharpest_head" in result
        assert "most_diffuse_head" in result
        assert "entropy_by_layer" in result
        assert "max_possible_entropy" in result

    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_entropy_profile(model, tokens)
        assert result["entropy_matrix"].shape == (2, 4)
        assert len(result["entropy_by_layer"]) == 2

    def test_entropy_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_entropy_profile(model, tokens)
        assert np.all(result["entropy_matrix"] >= -1e-8)


class TestPositionalAttentionBias:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = positional_attention_bias(model, tokens)
        assert "bos_attention" in result
        assert "recency_bias" in result
        assert "diagonal_strength" in result
        assert "mean_attention_distance" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = positional_attention_bias(model, tokens)
        assert result["bos_attention"].shape == (2, 4)
        assert result["recency_bias"].shape == (2, 4)

    def test_bos_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = positional_attention_bias(model, tokens)
        assert np.all(result["bos_attention"] >= -1e-5)
        assert np.all(result["bos_attention"] <= 1.0 + 1e-5)


class TestAttentionSparsityAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_sparsity_analysis(model, tokens)
        assert "sparsity_matrix" in result
        assert "mean_tokens_attended" in result
        assert "sparsest_head" in result
        assert "densest_head" in result
        assert "gini_coefficients" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_sparsity_analysis(model, tokens)
        assert result["sparsity_matrix"].shape == (2, 4)
        assert result["gini_coefficients"].shape == (2, 4)

    def test_sparsity_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_sparsity_analysis(model, tokens)
        assert np.all(result["sparsity_matrix"] >= 0)
        assert np.all(result["sparsity_matrix"] <= 1.0 + 1e-5)


class TestCrossHeadAttentionSimilarity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_head_attention_similarity(model, tokens)
        assert "within_layer_similarity" in result
        assert "across_layer_similarity" in result
        assert "most_similar_pair" in result
        assert "most_dissimilar_pair" in result
        assert "redundancy_score" in result

    def test_within_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_head_attention_similarity(model, tokens)
        assert len(result["within_layer_similarity"]) == 2

    def test_redundancy_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_head_attention_similarity(model, tokens)
        assert 0.0 <= result["redundancy_score"] <= 1.0


class TestAttentionHeadClassification:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_head_classification(model, tokens)
        assert "classifications" in result
        assert "class_counts" in result
        assert "confidence_scores" in result

    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_head_classification(model, tokens)
        assert result["classifications"].shape == (2, 4)
        assert result["confidence_scores"].shape == (2, 4)

    def test_valid_classes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_head_classification(model, tokens)
        valid = {"positional", "content", "previous_token", "uniform"}
        for c in result["classifications"].flatten():
            assert c in valid
