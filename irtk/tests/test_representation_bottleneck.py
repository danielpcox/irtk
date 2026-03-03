"""Tests for representation_bottleneck module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.representation_bottleneck import (
    layer_compression_analysis,
    representational_capacity,
    redundancy_analysis,
    information_flow_bottleneck,
    cross_position_redundancy,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


@pytest.fixture
def tokens_list():
    return [
        jnp.array([0, 5, 10, 15, 20, 25, 30, 35]),
        jnp.array([1, 6, 11, 16, 21, 26, 31, 36]),
        jnp.array([2, 7, 12, 17, 22, 27, 32, 37]),
        jnp.array([3, 8, 13, 18, 23, 28, 33, 38]),
        jnp.array([4, 9, 14, 19, 24, 29, 34, 39]),
    ]


class TestLayerCompressionAnalysis:
    def test_output_keys(self, model, tokens_list):
        r = layer_compression_analysis(model, tokens_list)
        assert "effective_dims" in r
        assert "compression_ratio" in r
        assert "bottleneck_layer" in r
        assert "expansion_layer" in r
        assert "total_compression" in r

    def test_shapes(self, model, tokens_list):
        r = layer_compression_analysis(model, tokens_list)
        n_layers = model.cfg.n_layers
        assert r["effective_dims"].shape == (n_layers + 1,)
        assert r["compression_ratio"].shape == (n_layers,)

    def test_dims_positive(self, model, tokens_list):
        r = layer_compression_analysis(model, tokens_list)
        assert np.all(r["effective_dims"] > 0)

    def test_total_compression_positive(self, model, tokens_list):
        r = layer_compression_analysis(model, tokens_list)
        assert r["total_compression"] > 0


class TestRepresentationalCapacity:
    def test_output_keys(self, model, tokens):
        r = representational_capacity(model, tokens)
        assert "utilization" in r
        assert "top_sv_fraction" in r
        assert "capacity_bits" in r
        assert "most_utilized_layer" in r
        assert "least_utilized_layer" in r

    def test_shapes(self, model, tokens):
        r = representational_capacity(model, tokens)
        n_layers = model.cfg.n_layers
        assert r["utilization"].shape == (n_layers + 1,)
        assert r["capacity_bits"].shape == (n_layers + 1,)

    def test_utilization_bounded(self, model, tokens):
        r = representational_capacity(model, tokens)
        assert np.all(r["utilization"] >= 0)
        assert np.all(r["utilization"] <= 1.0 + 1e-5)


class TestRedundancyAnalysis:
    def test_output_keys(self, model, tokens):
        r = redundancy_analysis(model, tokens)
        assert "cosine_similarities" in r
        assert "residual_norms" in r
        assert "relative_change" in r
        assert "most_redundant_layer" in r
        assert "most_transformative_layer" in r

    def test_shapes(self, model, tokens):
        r = redundancy_analysis(model, tokens)
        n_layers = model.cfg.n_layers
        assert r["cosine_similarities"].shape == (n_layers,)
        assert r["residual_norms"].shape == (n_layers,)

    def test_similarity_bounded(self, model, tokens):
        r = redundancy_analysis(model, tokens)
        assert np.all(r["cosine_similarities"] >= -1.0 - 1e-5)
        assert np.all(r["cosine_similarities"] <= 1.0 + 1e-5)

    def test_norms_nonneg(self, model, tokens):
        r = redundancy_analysis(model, tokens)
        assert np.all(r["residual_norms"] >= 0)


class TestInformationFlowBottleneck:
    def test_output_keys(self, model, tokens):
        r = information_flow_bottleneck(model, tokens)
        assert "attn_info_fraction" in r
        assert "mlp_info_fraction" in r
        assert "attn_norms" in r
        assert "mlp_norms" in r
        assert "bottleneck_layer" in r
        assert "dominant_pathway" in r

    def test_shapes(self, model, tokens):
        r = information_flow_bottleneck(model, tokens)
        n_layers = model.cfg.n_layers
        assert r["attn_info_fraction"].shape == (n_layers,)
        assert r["mlp_info_fraction"].shape == (n_layers,)

    def test_fractions_sum_to_one(self, model, tokens):
        r = information_flow_bottleneck(model, tokens)
        for l in range(model.cfg.n_layers):
            assert abs(r["attn_info_fraction"][l] + r["mlp_info_fraction"][l] - 1.0) < 1e-4

    def test_dominant_valid(self, model, tokens):
        r = information_flow_bottleneck(model, tokens)
        assert r["dominant_pathway"] in ("attention", "mlp")


class TestCrossPositionRedundancy:
    def test_output_keys(self, model, tokens):
        r = cross_position_redundancy(model, tokens)
        assert "mean_pairwise_similarity" in r
        assert "position_effective_rank" in r
        assert "most_redundant_layer" in r
        assert "most_diverse_layer" in r

    def test_shapes(self, model, tokens):
        r = cross_position_redundancy(model, tokens)
        n_layers = model.cfg.n_layers
        assert r["mean_pairwise_similarity"].shape == (n_layers + 1,)
        assert r["position_effective_rank"].shape == (n_layers + 1,)

    def test_rank_positive(self, model, tokens):
        r = cross_position_redundancy(model, tokens)
        assert np.all(r["position_effective_rank"] > 0)
