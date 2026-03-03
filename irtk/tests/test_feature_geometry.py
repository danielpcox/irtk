"""Tests for SAE feature dictionary geometry analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.feature_geometry import (
    feature_splitting_analysis,
    feature_absorption_detection,
    feature_universality,
    feature_interaction_graph,
    decoder_geometry_stats,
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


def _make_sae(d_model=16, n_features=32, seed=99):
    key = jax.random.PRNGKey(seed)
    return SparseAutoencoder(d_model, n_features, key=key)


# ─── Feature Splitting Analysis ──────────────────────────────────────────────


class TestFeatureSplittingAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        sae_s = _make_sae(n_features=16, seed=42)
        sae_l = _make_sae(n_features=32, seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_splitting_analysis(sae_s, sae_l, model, seqs, "blocks.0.hook_resid_post")
        assert "split_features" in result
        assert "unsplit_features" in result
        assert "decoder_cosines" in result

    def test_cosines_shape(self):
        sae_s = _make_sae(n_features=16, seed=42)
        sae_l = _make_sae(n_features=32, seed=99)
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_splitting_analysis(sae_s, sae_l, model, seqs, "blocks.0.hook_resid_post")
        assert result["decoder_cosines"].shape == (16, 32)

    def test_split_count_positive(self):
        sae_s = _make_sae(n_features=16, seed=42)
        sae_l = _make_sae(n_features=32, seed=99)
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_splitting_analysis(sae_s, sae_l, model, seqs, "blocks.0.hook_resid_post")
        assert result["mean_split_count"] >= 1.0


# ─── Feature Absorption Detection ────────────────────────────────────────────


class TestFeatureAbsorptionDetection:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = feature_absorption_detection(sae, model, seqs, "blocks.0.hook_resid_post", 0, 1)
        assert "absorption_score" in result
        assert "a_rate_without_b" in result
        assert "suppression_ratio" in result

    def test_absorption_in_range(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = feature_absorption_detection(sae, model, seqs, "blocks.0.hook_resid_post", 0, 1)
        assert 0 <= result["absorption_score"] <= 1.0

    def test_rates_non_negative(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = feature_absorption_detection(sae, model, seqs, "blocks.0.hook_resid_post", 0, 1)
        assert result["a_rate_without_b"] >= 0
        assert result["a_rate_with_b"] >= 0


# ─── Feature Universality ────────────────────────────────────────────────────


class TestFeatureUniversality:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        sae_a = _make_sae(seed=42)
        sae_b = _make_sae(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_universality(sae_a, sae_b, model_a, model_b, seqs, "blocks.0.hook_resid_post")
        assert "matched_pairs" in result
        assert "universality_rate" in result
        assert "mean_match_correlation" in result

    def test_universality_in_range(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        sae_a = _make_sae(seed=42)
        sae_b = _make_sae(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_universality(sae_a, sae_b, model_a, model_b, seqs, "blocks.0.hook_resid_post")
        assert 0 <= result["universality_rate"] <= 1.0

    def test_same_model_higher_universality(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = feature_universality(sae, sae, model, model, seqs, "blocks.0.hook_resid_post")
        assert result["mean_match_correlation"] >= 0


# ─── Feature Interaction Graph ────────────────────────────────────────────────


class TestFeatureInteractionGraph:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_interaction_graph(sae, model, seqs, "blocks.0.hook_resid_post")
        assert "co_occurrence" in result
        assert "correlations" in result
        assert "suppression_pairs" in result
        assert "feature_indices" in result

    def test_matrices_square(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_interaction_graph(sae, model, seqs, "blocks.0.hook_resid_post", top_k=10)
        if result["co_occurrence"].size > 0:
            assert result["co_occurrence"].shape[0] == result["co_occurrence"].shape[1]

    def test_suppression_pairs_valid(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_interaction_graph(sae, model, seqs, "blocks.0.hook_resid_post")
        for i, j in result["suppression_pairs"]:
            assert isinstance(i, int)
            assert isinstance(j, int)


# ─── Decoder Geometry Stats ──────────────────────────────────────────────────


class TestDecoderGeometryStats:
    def test_returns_dict(self):
        sae = _make_sae()
        result = decoder_geometry_stats(sae)
        assert "pairwise_cosines" in result
        assert "near_duplicates" in result
        assert "feature_norms" in result
        assert "mean_pairwise_cosine" in result

    def test_cosines_shape(self):
        sae = _make_sae(n_features=16)
        result = decoder_geometry_stats(sae)
        assert result["pairwise_cosines"].shape == (16, 16)

    def test_norms_length(self):
        sae = _make_sae(n_features=16)
        result = decoder_geometry_stats(sae)
        assert len(result["feature_norms"]) == 16

    def test_diagonal_ones(self):
        sae = _make_sae()
        result = decoder_geometry_stats(sae)
        diag = np.diag(result["pairwise_cosines"])
        np.testing.assert_allclose(diag, 1.0, atol=0.01)
