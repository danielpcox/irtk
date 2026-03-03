"""Tests for polysemanticity detection and analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.polysemantic_features import (
    polysemanticity_score,
    feature_context_clusters,
    activation_decomposition,
    feature_interference_matrix,
    monosemanticity_ranking,
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


# ─── Polysemanticity Score ───────────────────────────────────────────────


class TestPolysemanticityScoree:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = polysemanticity_score(model, "blocks.0.hook_resid_post", seqs, 0)
        assert "score" in result
        assert "activation_variance" in result
        assert "bimodality" in result
        assert "firing_rate" in result

    def test_score_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = polysemanticity_score(model, "blocks.0.hook_resid_post", seqs, 0)
        assert result["score"] >= 0

    def test_firing_rate_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = polysemanticity_score(model, "blocks.0.hook_resid_post", seqs, 0)
        assert 0 <= result["firing_rate"] <= 1.0

    def test_variance_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = polysemanticity_score(model, "blocks.0.hook_resid_post", seqs, 0)
        assert result["activation_variance"] >= 0

    def test_invalid_hook(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = polysemanticity_score(model, "nonexistent.hook", seqs, 0)
        assert result["score"] == 0.0


# ─── Feature Context Clusters ────────────────────────────────────────────


class TestFeatureContextClusters:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5, 6, 7])]
        result = feature_context_clusters(model, "blocks.0.hook_resid_post", seqs, 0, n_clusters=2, top_k=10)
        assert "cluster_assignments" in result
        assert "cluster_sizes" in result
        assert "n_effective_clusters" in result
        assert "activation_values" in result

    def test_n_effective_bounded(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5, 6, 7])]
        result = feature_context_clusters(model, "blocks.0.hook_resid_post", seqs, 0, n_clusters=3, top_k=20)
        assert 0 <= result["n_effective_clusters"] <= 3

    def test_cluster_sizes_sum(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5, 6, 7])]
        result = feature_context_clusters(model, "blocks.0.hook_resid_post", seqs, 0, n_clusters=2, top_k=10)
        if len(result["cluster_assignments"]) > 0:
            assert int(np.sum(result["cluster_sizes"])) == len(result["cluster_assignments"])

    def test_empty_on_bad_hook(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_context_clusters(model, "nonexistent.hook", seqs, 0)
        assert len(result["cluster_assignments"]) == 0


# ─── Activation Decomposition ────────────────────────────────────────────


class TestActivationDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_decomposition(sae, model, tokens, "blocks.0.hook_resid_post")
        assert "active_features" in result
        assert "n_active" in result
        assert "reconstruction_error" in result
        assert "top_feature_fraction" in result

    def test_n_active_bounded(self):
        model = _make_model()
        sae = _make_sae(n_features=32)
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_decomposition(sae, model, tokens, "blocks.0.hook_resid_post")
        assert 0 <= result["n_active"] <= 32

    def test_recon_error_non_negative(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_decomposition(sae, model, tokens, "blocks.0.hook_resid_post")
        assert result["reconstruction_error"] >= 0

    def test_features_sorted_descending(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_decomposition(sae, model, tokens, "blocks.0.hook_resid_post")
        if len(result["active_features"]) > 1:
            vals = [v for _, v in result["active_features"]]
            assert vals == sorted(vals, reverse=True)

    def test_bad_hook_returns_empty(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = activation_decomposition(sae, model, tokens, "nonexistent")
        assert result["n_active"] == 0


# ─── Feature Interference Matrix ─────────────────────────────────────────


class TestFeatureInterferenceMatrix:
    def test_returns_dict(self):
        sae = _make_sae()
        result = feature_interference_matrix(sae, [0, 1, 2])
        assert "interference_matrix" in result
        assert "max_interference" in result
        assert "mean_interference" in result
        assert "orthogonality_score" in result

    def test_matrix_shape(self):
        sae = _make_sae()
        result = feature_interference_matrix(sae, [0, 1, 2, 3])
        assert result["interference_matrix"].shape == (4, 4)

    def test_diagonal_ones(self):
        sae = _make_sae()
        result = feature_interference_matrix(sae, [0, 1, 2])
        diag = np.diag(result["interference_matrix"])
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)

    def test_orthogonality_in_range(self):
        sae = _make_sae()
        result = feature_interference_matrix(sae, [0, 1, 2])
        assert 0 <= result["orthogonality_score"] <= 1.0

    def test_single_feature(self):
        sae = _make_sae()
        result = feature_interference_matrix(sae, [0])
        assert result["max_interference"] == 0.0
        assert result["mean_interference"] == 0.0


# ─── Monosemanticity Ranking ─────────────────────────────────────────────


class TestMonosemantcityRanking:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = monosemanticity_ranking(model, "blocks.0.hook_resid_post", seqs, top_k=5)
        assert "most_monosemantic" in result
        assert "most_polysemantic" in result
        assert "all_scores" in result
        assert "mean_score" in result

    def test_top_k_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = monosemanticity_ranking(model, "blocks.0.hook_resid_post", seqs, top_k=5)
        assert len(result["most_monosemantic"]) <= 5
        assert len(result["most_polysemantic"]) <= 5

    def test_all_scores_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = monosemanticity_ranking(model, "blocks.0.hook_resid_post", seqs)
        assert len(result["all_scores"]) == model.cfg.d_model

    def test_scores_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = monosemanticity_ranking(model, "blocks.0.hook_resid_post", seqs)
        assert np.all(result["all_scores"] >= 0)

    def test_empty_on_bad_hook(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = monosemanticity_ranking(model, "nonexistent.hook", seqs)
        assert len(result["all_scores"]) == 0
