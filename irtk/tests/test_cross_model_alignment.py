"""Tests for cross-model structural alignment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.cross_model_alignment import (
    cka_layer_correspondence,
    match_heads_across_models,
    circuit_universality_score,
    aligned_feature_comparison,
    scale_law_trajectory,
)


def _make_model(seed=42, n_layers=2, d_model=16, n_heads=4):
    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=d_model, n_ctx=32,
        d_head=d_model // n_heads, n_heads=n_heads, d_vocab=50,
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


# ─── CKA Layer Correspondence ───────────────────────────────────────────────


class TestCKALayerCorrespondence:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        tokens = jnp.array([0, 1, 2, 3])
        result = cka_layer_correspondence(model_a, model_b, tokens)
        assert "cka_matrix" in result
        assert "best_match" in result
        assert "mean_cka" in result

    def test_matrix_shape(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        tokens = jnp.array([0, 1, 2, 3])
        result = cka_layer_correspondence(model_a, model_b, tokens)
        assert result["cka_matrix"].shape == (2, 2)

    def test_self_similarity_high(self):
        model = _make_model(seed=42)
        tokens = jnp.array([0, 1, 2, 3])
        result = cka_layer_correspondence(model, model, tokens)
        # Same model should have high diagonal CKA
        for i in range(2):
            assert result["cka_matrix"][i, i] > 0.5


# ─── Match Heads Across Models ──────────────────────────────────────────────


class TestMatchHeadsAcrossModels:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        tokens = jnp.array([0, 1, 2, 3])
        result = match_heads_across_models(model_a, model_b, tokens)
        assert "matches" in result
        assert "similarity_matrix" in result
        assert "best_match_per_head_a" in result

    def test_matches_length(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        tokens = jnp.array([0, 1, 2, 3])
        result = match_heads_across_models(model_a, model_b, tokens)
        assert len(result["matches"]) == 8  # 2 layers * 4 heads

    def test_ov_cosine_metric(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        tokens = jnp.array([0, 1, 2, 3])
        result = match_heads_across_models(
            model_a, model_b, tokens, metric="ov_cosine"
        )
        assert "matches" in result


# ─── Circuit Universality Score ──────────────────────────────────────────────


class TestCircuitUniversalityScore:
    def test_returns_dict(self):
        models = [_make_model(seed=i) for i in range(3)]
        tokens = jnp.array([0, 1, 2, 3])
        result = circuit_universality_score(
            [(0, 0), (1, 0)], models, tokens, _metric
        )
        assert "per_model_faithfulness" in result
        assert "universality_score" in result
        assert len(result["per_model_faithfulness"]) == 3

    def test_empty_inputs(self):
        result = circuit_universality_score([], [], jnp.array([0, 1]), _metric)
        assert result["universality_score"] == 0.0


# ─── Aligned Feature Comparison ─────────────────────────────────────────────


class TestAlignedFeatureComparison:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        sae_a = SparseAutoencoder(16, 32, key=jax.random.PRNGKey(0))
        sae_b = SparseAutoencoder(16, 32, key=jax.random.PRNGKey(1))
        tokens = jnp.array([0, 1, 2, 3])
        result = aligned_feature_comparison(
            sae_a, sae_b, model_a, model_b, tokens,
            "blocks.0.hook_resid_post", "blocks.0.hook_resid_post"
        )
        assert "matched_pairs" in result
        assert "activation_correlation" in result
        assert "mean_similarity" in result


# ─── Scale Law Trajectory ───────────────────────────────────────────────────


class TestScaleLawTrajectory:
    def test_returns_dict(self):
        models = [_make_model(seed=i) for i in range(3)]
        sizes = [100, 200, 400]
        tokens = jnp.array([0, 1, 2, 3])
        result = scale_law_trajectory(models, sizes, tokens, _metric)
        assert "sizes" in result
        assert "metrics" in result
        assert "trend" in result
        assert "log_log_slope" in result

    def test_sorted_output(self):
        models = [_make_model(seed=i) for i in range(3)]
        sizes = [400, 100, 200]  # Unsorted
        tokens = jnp.array([0, 1, 2, 3])
        result = scale_law_trajectory(models, sizes, tokens, _metric)
        assert result["sizes"] == [100, 200, 400]

    def test_empty_inputs(self):
        result = scale_law_trajectory([], [], jnp.array([0, 1]), _metric)
        assert result["trend"] == "flat"
