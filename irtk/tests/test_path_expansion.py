"""Tests for path expansion analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.path_expansion import (
    enumerate_paths,
    path_contribution_matrix,
    virtual_weight_path,
    residual_stream_decomposition,
    path_patching_matrix,
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


def _metric(logits):
    return float(logits[-1, 0])


class TestEnumeratePaths:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = enumerate_paths(model, tokens, max_depth=2, top_k=3)
        assert "paths" in result
        assert "path_contributions" in result
        assert "n_paths_enumerated" in result
        assert "top_paths" in result

    def test_top_paths_count(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = enumerate_paths(model, tokens, max_depth=2, top_k=3)
        assert len(result["top_paths"]) <= 3

    def test_contributions_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = enumerate_paths(model, tokens, max_depth=2, top_k=5)
        assert np.all(result["path_contributions"] >= 0)

    def test_paths_enumerated(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = enumerate_paths(model, tokens, max_depth=1, top_k=5)
        # Should have at least n_layers * n_heads + n_layers paths (depth 1)
        assert result["n_paths_enumerated"] >= 2 * 4 + 2


class TestPathContributionMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_contribution_matrix(model, tokens)
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "embed_contribution" in result
        assert "total_contribution" in result
        assert "dominant_component" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_contribution_matrix(model, tokens)
        assert result["attn_contributions"].shape == (2, 4)
        assert result["mlp_contributions"].shape == (2,)

    def test_dominant_is_tuple(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_contribution_matrix(model, tokens)
        assert isinstance(result["dominant_component"], tuple)


class TestVirtualWeightPath:
    def test_returns_dict(self):
        model = _make_model()
        result = virtual_weight_path(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        assert "ov_ov_composition" in result
        assert "ov_qk_composition" in result
        assert "composition_score" in result
        assert "ov_a_matrix" in result

    def test_norms_nonneg(self):
        model = _make_model()
        result = virtual_weight_path(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        assert result["ov_ov_composition"] >= 0
        assert result["ov_qk_composition"] >= 0
        assert result["composition_score"] >= 0

    def test_ov_matrix_shape(self):
        model = _make_model()
        result = virtual_weight_path(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        assert result["ov_a_matrix"].shape == (16, 16)


class TestResidualStreamDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_decomposition(model, tokens)
        assert "cumulative_norms" in result
        assert "attn_added_norms" in result
        assert "mlp_added_norms" in result
        assert "growth_rate" in result
        assert "embed_norm" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_decomposition(model, tokens)
        assert len(result["cumulative_norms"]) == 3  # n_layers + 1
        assert len(result["attn_added_norms"]) == 2
        assert len(result["mlp_added_norms"]) == 2

    def test_norms_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_decomposition(model, tokens)
        assert np.all(result["cumulative_norms"] >= 0)
        assert np.all(result["attn_added_norms"] >= 0)
        assert np.all(result["mlp_added_norms"] >= 0)


class TestPathPatchingMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = path_patching_matrix(model, tokens, corrupted, _metric)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "baseline_metric" in result
        assert "corrupted_metric" in result
        assert "most_important_component" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = path_patching_matrix(model, tokens, corrupted, _metric)
        assert result["attn_effects"].shape == (2, 4)
        assert result["mlp_effects"].shape == (2,)
