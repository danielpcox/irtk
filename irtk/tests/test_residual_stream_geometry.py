"""Tests for residual_stream_geometry module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_geometry import (
    residual_angle_structure, residual_subspace_dimension,
    residual_update_geometry, residual_pairwise_distances,
    residual_geometry_summary,
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
    return jnp.array([1, 5, 10, 15, 20])


def test_angle_structure(model, tokens):
    result = residual_angle_structure(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_angle_range(model, tokens):
    result = residual_angle_structure(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["cosine_to_embed"] <= 1
        assert -1 <= p["cosine_to_unembed"] <= 1
        assert p["norm"] > 0


def test_subspace_dimension_structure(model, tokens):
    result = residual_subspace_dimension(model, tokens, layer=0)
    assert result["participation_ratio"] > 0
    assert result["dim_for_90_pct"] >= 1


def test_subspace_dimension_sv(model, tokens):
    result = residual_subspace_dimension(model, tokens, layer=0)
    assert 0 <= result["top_sv_fraction"] <= 1


def test_update_geometry_structure(model, tokens):
    result = residual_update_geometry(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["mostly_perpendicular"] >= 0


def test_update_geometry_components(model, tokens):
    result = residual_update_geometry(model, tokens)
    for p in result["per_layer"]:
        assert p["update_norm"] >= 0
        assert p["perpendicular_magnitude"] >= 0


def test_pairwise_distances_structure(model, tokens):
    result = residual_pairwise_distances(model, tokens, layer=0)
    assert result["distance_matrix"].shape == (5, 5)
    assert result["mean_distance"] >= 0


def test_pairwise_distances_symmetric(model, tokens):
    result = residual_pairwise_distances(model, tokens, layer=0)
    mat = result["distance_matrix"]
    assert float(jnp.max(jnp.abs(mat - mat.T))) < 1e-5


def test_geometry_summary_structure(model, tokens):
    result = residual_geometry_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["norm_trend"] in ("growing", "shrinking")
    for p in result["per_layer"]:
        assert p["mean_norm"] > 0
        assert p["participation_ratio"] > 0
