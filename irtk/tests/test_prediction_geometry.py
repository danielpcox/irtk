"""Tests for prediction geometry analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.prediction_geometry import (
    vocab_projection_trajectory,
    prediction_sharpening_rate,
    unembedding_alignment_per_head,
    token_promotion_geometry,
    final_layer_residual_decomposition,
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


class TestVocabProjectionTrajectory:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = vocab_projection_trajectory(model, tokens)
        assert "layer_top_tokens" in result
        assert "layer_top_logits" in result
        assert "final_enters_top_k_layer" in result
        assert "final_becomes_top_1_layer" in result
        assert "final_prediction" in result

    def test_layers_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = vocab_projection_trajectory(model, tokens, top_k=3)
        assert len(result["layer_top_tokens"]) == 2
        for tops in result["layer_top_tokens"]:
            assert len(tops) == 3

    def test_final_prediction_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = vocab_projection_trajectory(model, tokens)
        assert 0 <= result["final_prediction"] < 50


class TestPredictionSharpeningRate:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_sharpening_rate(model, tokens)
        assert "layer_entropies" in result
        assert "layer_top1_probs" in result
        assert "sharpening_rates" in result
        assert "crystallization_layer" in result
        assert "total_sharpening" in result

    def test_entropies_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_sharpening_rate(model, tokens)
        assert np.all(result["layer_entropies"] >= 0)

    def test_top1_probs_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = prediction_sharpening_rate(model, tokens)
        assert np.all(result["layer_top1_probs"] >= 0)
        assert np.all(result["layer_top1_probs"] <= 1.0)


class TestUnembeddingAlignmentPerHead:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = unembedding_alignment_per_head(model, tokens, layer=0)
        assert "promoted_tokens" in result
        assert "promoted_logits" in result
        assert "demoted_tokens" in result
        assert "demoted_logits" in result
        assert "head_logit_norms" in result

    def test_heads_count(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = unembedding_alignment_per_head(model, tokens, layer=0, top_k=3)
        assert len(result["promoted_tokens"]) == 4  # n_heads
        for promoted in result["promoted_tokens"]:
            assert len(promoted) == 3

    def test_norms_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = unembedding_alignment_per_head(model, tokens, layer=0)
        assert np.all(result["head_logit_norms"] >= 0)


class TestTokenPromotionGeometry:
    def test_returns_dict(self):
        model = _make_model()
        result = token_promotion_geometry(model, [0, 1, 2, 3])
        assert "pairwise_similarity" in result
        assert "mean_pairwise_similarity" in result
        assert "norms" in result
        assert "mean_cosine_to_centroid" in result

    def test_matrix_shape(self):
        model = _make_model()
        result = token_promotion_geometry(model, [0, 1, 2])
        assert result["pairwise_similarity"].shape == (3, 3)

    def test_diagonal_ones(self):
        model = _make_model()
        result = token_promotion_geometry(model, [0, 1, 2])
        for i in range(3):
            assert np.isclose(result["pairwise_similarity"][i, i], 1.0, atol=0.01)

    def test_similarity_bounded(self):
        model = _make_model()
        result = token_promotion_geometry(model, [0, 1, 2])
        assert np.all(result["pairwise_similarity"] >= -1.0 - 1e-5)
        assert np.all(result["pairwise_similarity"] <= 1.0 + 1e-5)


class TestFinalLayerResidualDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = final_layer_residual_decomposition(model, tokens)
        assert "parallel_contributions" in result
        assert "orthogonal_norms" in result
        assert "prediction_token" in result
        assert "total_parallel" in result
        assert "dominant_layer" in result

    def test_parallel_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = final_layer_residual_decomposition(model, tokens)
        # n_layers + 1 (embedding + each layer)
        assert len(result["parallel_contributions"]) == 3

    def test_orthogonal_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = final_layer_residual_decomposition(model, tokens)
        assert np.all(result["orthogonal_norms"] >= 0)
