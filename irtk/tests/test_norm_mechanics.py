"""Tests for LayerNorm mechanics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.norm_mechanics import (
    feature_scaling_by_norm,
    norm_directionality_bias,
    pre_vs_post_norm_effect,
    norm_gradient_flow,
    feature_whitening_by_layer,
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


class TestFeatureScalingByNorm:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_scaling_by_norm(model, tokens, layer=0)
        assert "scaling_factors" in result
        assert "most_amplified" in result
        assert "mean_scaling" in result

    def test_scaling_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_scaling_by_norm(model, tokens, layer=0)
        assert len(result["scaling_factors"]) == 16


class TestNormDirectionalityBias:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = norm_directionality_bias(model, tokens, layer=0)
        assert "weight_direction" in result
        assert "anisotropy" in result

    def test_weight_direction_unit_norm(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = norm_directionality_bias(model, tokens, layer=0)
        norm = np.linalg.norm(result["weight_direction"])
        np.testing.assert_allclose(norm, 1.0, atol=0.01)


class TestPreVsPostNormEffect:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pre_vs_post_norm_effect(model, tokens, layer=0)
        assert "cosine_similarity" in result
        assert "norm_ratio" in result
        assert "rank_pre" in result

    def test_cosine_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pre_vs_post_norm_effect(model, tokens, layer=0)
        assert -1.01 <= result["cosine_similarity"] <= 1.01


class TestNormGradientFlow:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = norm_gradient_flow(model, tokens, layer=0)
        assert "pre_norm_grad_norms" in result
        assert "post_norm_grad_norms" in result
        assert "gradient_amplification" in result

    def test_grad_norms_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = norm_gradient_flow(model, tokens, layer=0)
        assert len(result["pre_norm_grad_norms"]) == 16


class TestFeatureWhiteningByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_whitening_by_layer(model, tokens)
        assert "pre_correlations" in result
        assert "post_correlations" in result
        assert "whitening_effect" in result

    def test_arrays_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_whitening_by_layer(model, tokens)
        assert len(result["pre_correlations"]) == 2
        assert len(result["post_correlations"]) == 2
