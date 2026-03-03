"""Tests for ov_circuit_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.ov_circuit_analysis import (
    ov_eigenvalue_decomposition,
    token_copying_strength,
    ov_semantic_role,
    ov_composition_between_layers,
    ov_unembedding_projection,
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


class TestOVEigenvalueDecomposition:
    def test_basic(self, model):
        result = ov_eigenvalue_decomposition(model, layer=0, head=0)
        assert "eigenvalues" in result
        assert "ov_trace" in result
        assert "ov_rank" in result
        assert "singular_values" in result

    def test_eigenvalue_count(self, model):
        result = ov_eigenvalue_decomposition(model, layer=0, head=0, top_k=3)
        assert len(result["eigenvalues"]) == 3

    def test_rank_positive(self, model):
        result = ov_eigenvalue_decomposition(model, layer=0, head=0)
        assert result["ov_rank"] >= 0


class TestTokenCopyingStrength:
    def test_basic(self, model):
        result = token_copying_strength(model, layer=0, head=0)
        assert "copy_diagonal" in result
        assert "mean_copy_strength" in result
        assert "top_copied_tokens" in result
        assert "copy_vs_suppress_ratio" in result

    def test_diagonal_shape(self, model):
        result = token_copying_strength(model, layer=0, head=0)
        assert result["copy_diagonal"].shape == (model.cfg.d_vocab,)

    def test_ratio_positive(self, model):
        result = token_copying_strength(model, layer=0, head=0)
        assert result["copy_vs_suppress_ratio"] >= 0


class TestOVSemanticRole:
    def test_basic(self, model, tokens):
        result = ov_semantic_role(model, tokens, layer=0, head=0)
        assert "ov_output" in result
        assert "output_norm" in result
        assert "top_promoted_tokens" in result
        assert "source_position_contributions" in result

    def test_output_shape(self, model, tokens):
        result = ov_semantic_role(model, tokens, layer=0, head=0)
        assert result["ov_output"].shape == (model.cfg.d_model,)

    def test_norm_nonneg(self, model, tokens):
        result = ov_semantic_role(model, tokens, layer=0, head=0)
        assert result["output_norm"] >= 0


class TestOVCompositionBetweenLayers:
    def test_basic(self, model):
        result = ov_composition_between_layers(model, 0, 0, 1, 0)
        assert "v_composition_score" in result
        assert "k_composition_score" in result
        assert "q_composition_score" in result
        assert "composed_ov_rank" in result

    def test_scores_nonneg(self, model):
        result = ov_composition_between_layers(model, 0, 0, 1, 0)
        assert result["v_composition_score"] >= 0
        assert result["k_composition_score"] >= 0
        assert result["q_composition_score"] >= 0


class TestOVUnembeddingProjection:
    def test_basic(self, model):
        result = ov_unembedding_projection(model, layer=0, head=0)
        assert "ov_logit_matrix_rank" in result
        assert "top_positive_directions" in result
        assert "top_negative_directions" in result
        assert "projection_norm" in result

    def test_rank_positive(self, model):
        result = ov_unembedding_projection(model, layer=0, head=0)
        assert result["ov_logit_matrix_rank"] >= 0

    def test_norm_positive(self, model):
        result = ov_unembedding_projection(model, layer=0, head=0)
        assert result["projection_norm"] >= 0
