"""Tests for low-rank analysis and approximation tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.low_rank import (
    weight_svd,
    effective_rank,
    low_rank_approximation,
    weight_spectrum_similarity,
    apply_low_rank_weights,
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


# ─── Weight SVD ─────────────────────────────────────────────────────────────


class TestWeightSVD:
    def test_returns_dict(self):
        model = _make_model()
        result = weight_svd(model, "blocks.0.mlp.W_in")
        assert isinstance(result, dict)
        assert "U" in result
        assert "S" in result
        assert "Vh" in result

    def test_singular_values_descending(self):
        model = _make_model()
        result = weight_svd(model, "blocks.0.mlp.W_in")
        S = result["S"]
        assert np.all(S[:-1] >= S[1:] - 1e-7)

    def test_rank_positive(self):
        model = _make_model()
        result = weight_svd(model, "blocks.0.mlp.W_in")
        assert result["rank"] > 0

    def test_3d_weight(self):
        model = _make_model()
        result = weight_svd(model, "blocks.0.attn.W_Q")
        assert result["shape"] == (4, 16, 4)

    def test_reconstruction(self):
        model = _make_model()
        result = weight_svd(model, "blocks.0.mlp.W_in")
        U, S, Vh = result["U"], result["S"], result["Vh"]
        W_reconstructed = (U * S) @ Vh
        W_original = np.array(model.blocks[0].mlp.W_in)
        np.testing.assert_allclose(W_reconstructed, W_original, atol=1e-5)


# ─── Effective Rank ─────────────────────────────────────────────────────────


class TestEffectiveRank:
    def test_returns_dict(self):
        model = _make_model()
        result = effective_rank(model, "blocks.0.mlp.W_in")
        assert "effective_rank" in result
        assert "full_rank" in result

    def test_effective_leq_full(self):
        model = _make_model()
        result = effective_rank(model, "blocks.0.mlp.W_in")
        assert result["effective_rank"] <= result["full_rank"]

    def test_higher_threshold_higher_rank(self):
        model = _make_model()
        r90 = effective_rank(model, "blocks.0.mlp.W_in", threshold=0.90)
        r99 = effective_rank(model, "blocks.0.mlp.W_in", threshold=0.99)
        assert r90["effective_rank"] <= r99["effective_rank"]

    def test_cumulative_energy_monotonic(self):
        model = _make_model()
        result = effective_rank(model, "blocks.0.mlp.W_in")
        cum = result["cumulative_energy"]
        assert np.all(cum[:-1] <= cum[1:] + 1e-7)

    def test_cumulative_energy_ends_at_one(self):
        model = _make_model()
        result = effective_rank(model, "blocks.0.mlp.W_in")
        np.testing.assert_allclose(result["cumulative_energy"][-1], 1.0, atol=1e-6)


# ─── Low-Rank Approximation ────────────────────────────────────────────────


class TestLowRankApproximation:
    def test_returns_dict(self):
        model = _make_model()
        result = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=4)
        assert "approximation" in result
        assert "reconstruction_error" in result

    def test_shape_preserved(self):
        model = _make_model()
        result = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=4)
        W = np.array(model.blocks[0].mlp.W_in)
        assert result["approximation"].shape == W.shape

    def test_3d_shape_preserved(self):
        model = _make_model()
        result = low_rank_approximation(model, "blocks.0.attn.W_Q", rank=2)
        assert result["approximation"].shape == (4, 16, 4)

    def test_full_rank_zero_error(self):
        model = _make_model()
        svd = weight_svd(model, "blocks.0.mlp.W_in")
        full_rank = len(svd["S"])
        result = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=full_rank)
        assert result["relative_error"] < 1e-5
        assert result["energy_captured"] > 0.9999

    def test_lower_rank_higher_error(self):
        model = _make_model()
        r2 = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=2)
        r8 = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=8)
        assert r2["reconstruction_error"] >= r8["reconstruction_error"] - 1e-7

    def test_energy_captured_increases(self):
        model = _make_model()
        r2 = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=2)
        r8 = low_rank_approximation(model, "blocks.0.mlp.W_in", rank=8)
        assert r2["energy_captured"] <= r8["energy_captured"] + 1e-7


# ─── Weight Spectrum Similarity ─────────────────────────────────────────────


class TestWeightSpectrumSimilarity:
    def test_returns_dict(self):
        model = _make_model()
        result = weight_spectrum_similarity(
            model, "blocks.0.mlp.W_in", "blocks.1.mlp.W_in"
        )
        assert "spectral_similarity" in result

    def test_self_similarity_is_one(self):
        model = _make_model()
        result = weight_spectrum_similarity(
            model, "blocks.0.mlp.W_in", "blocks.0.mlp.W_in"
        )
        assert abs(result["spectral_similarity"] - 1.0) < 1e-5

    def test_similarity_between_zero_and_one(self):
        model = _make_model()
        result = weight_spectrum_similarity(
            model, "blocks.0.mlp.W_in", "blocks.1.mlp.W_in"
        )
        assert -0.01 <= result["spectral_similarity"] <= 1.01

    def test_different_weight_types(self):
        model = _make_model()
        result = weight_spectrum_similarity(
            model, "blocks.0.mlp.W_in", "blocks.0.mlp.W_out"
        )
        assert "rank_a" in result
        assert "rank_b" in result


# ─── Apply Low-Rank Weights ────────────────────────────────────────────────


class TestApplyLowRankWeights:
    def test_returns_model(self):
        model = _make_model()
        result = apply_low_rank_weights(model, "blocks.0.mlp.W_in", rank=4)
        assert isinstance(result, HookedTransformer)

    def test_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        modified = apply_low_rank_weights(model, "blocks.0.mlp.W_in", rank=2)
        logits_after = modified(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_original_unchanged(self):
        model = _make_model()
        W_orig = np.array(model.blocks[0].mlp.W_in)
        _ = apply_low_rank_weights(model, "blocks.0.mlp.W_in", rank=2)
        np.testing.assert_allclose(model.blocks[0].mlp.W_in, W_orig)

    def test_full_rank_preserves_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        svd = weight_svd(model, "blocks.0.mlp.W_in")
        full_rank = len(svd["S"])
        modified = apply_low_rank_weights(model, "blocks.0.mlp.W_in", rank=full_rank)
        logits_after = modified(tokens)
        np.testing.assert_allclose(logits_before, logits_after, atol=1e-4)
