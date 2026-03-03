"""Tests for weight_decomposition module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_decomposition import (
    weight_svd_decomposition,
    weight_shared_substructure,
    weight_clustering,
    spectral_weight_analysis,
    weight_norm_distribution,
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


class TestWeightSVDDecomposition:
    def test_attn(self, model):
        result = weight_svd_decomposition(model, layer=0, component="attn")
        assert "singular_values" in result
        assert "effective_ranks" in result
        assert "total_params" in result
        assert "compression_ratio" in result
        assert "W_Q" in result["singular_values"]
        assert "W_O" in result["singular_values"]

    def test_mlp(self, model):
        result = weight_svd_decomposition(model, layer=0, component="mlp")
        assert "W_in" in result["singular_values"]
        assert "W_out" in result["singular_values"]

    def test_compression_ratio(self, model):
        result = weight_svd_decomposition(model, layer=0, component="attn")
        assert 0 <= result["compression_ratio"] <= 1


class TestWeightSharedSubstructure:
    def test_basic(self, model):
        result = weight_shared_substructure(model)
        assert "cross_layer_similarity" in result
        assert "most_similar_layers" in result
        assert "shared_subspace_dim" in result
        assert "layer_weight_norms" in result

    def test_shapes(self, model):
        result = weight_shared_substructure(model)
        nl = model.cfg.n_layers
        assert result["cross_layer_similarity"].shape == (nl, nl)
        assert result["layer_weight_norms"].shape == (nl,)

    def test_self_similarity(self, model):
        result = weight_shared_substructure(model)
        diag = np.diag(result["cross_layer_similarity"])
        np.testing.assert_allclose(diag, 1.0, atol=0.01)


class TestWeightClustering:
    def test_basic(self, model):
        result = weight_clustering(model, n_clusters=2)
        assert "labels" in result
        assert "cluster_sizes" in result
        assert "within_cluster_variance" in result

    def test_all_assigned(self, model):
        result = weight_clustering(model, n_clusters=2)
        n_items = model.cfg.n_layers * 2  # attn + mlp per layer
        assert len(result["labels"]) == n_items
        assert sum(result["cluster_sizes"]) == n_items

    def test_variance_nonneg(self, model):
        result = weight_clustering(model, n_clusters=2)
        assert np.all(result["within_cluster_variance"] >= 0)


class TestSpectralWeightAnalysis:
    def test_basic(self, model):
        result = spectral_weight_analysis(model)
        assert "condition_numbers" in result
        assert "spectral_norms" in result
        assert "rank_deficiency" in result
        assert "worst_conditioned" in result

    def test_condition_positive(self, model):
        result = spectral_weight_analysis(model)
        for key, val in result["condition_numbers"].items():
            assert val > 0

    def test_spectral_norms_positive(self, model):
        result = spectral_weight_analysis(model)
        for key, val in result["spectral_norms"].items():
            assert val > 0


class TestWeightNormDistribution:
    def test_basic(self, model):
        result = weight_norm_distribution(model)
        assert "component_norms" in result
        assert "attn_norms_by_layer" in result
        assert "mlp_norms_by_layer" in result
        assert "total_norm" in result
        assert "norm_ratio_attn_mlp" in result

    def test_shapes(self, model):
        result = weight_norm_distribution(model)
        nl = model.cfg.n_layers
        assert result["attn_norms_by_layer"].shape == (nl,)
        assert result["mlp_norms_by_layer"].shape == (nl,)
        assert result["norm_ratio_attn_mlp"].shape == (nl,)

    def test_norms_positive(self, model):
        result = weight_norm_distribution(model)
        assert result["total_norm"] > 0
        assert np.all(result["attn_norms_by_layer"] > 0)
        assert np.all(result["mlp_norms_by_layer"] > 0)
