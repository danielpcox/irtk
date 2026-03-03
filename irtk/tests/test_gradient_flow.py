"""Tests for gradient flow analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.gradient_flow import (
    gradient_norm_by_layer,
    component_gradient_attribution,
    gradient_saturation_analysis,
    layernorm_gradient_effect,
    per_head_gradient_sensitivity,
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


class TestGradientNormByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_norm_by_layer(model, tokens)
        assert "layer_grad_norms" in result
        assert "max_grad_layer" in result
        assert "min_grad_layer" in result
        assert "gradient_ratio" in result
        assert "vanishing" in result

    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_norm_by_layer(model, tokens)
        assert len(result["layer_grad_norms"]) == 3  # n_layers + 1

    def test_norms_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_norm_by_layer(model, tokens)
        assert np.all(result["layer_grad_norms"] >= 0)


class TestComponentGradientAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = component_gradient_attribution(model, tokens)
        assert "attn_grad_norms" in result
        assert "mlp_grad_norms" in result
        assert "attn_fraction" in result
        assert "dominant_component_per_layer" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = component_gradient_attribution(model, tokens)
        assert result["attn_grad_norms"].shape == (2,)
        assert result["mlp_grad_norms"].shape == (2,)
        assert len(result["dominant_component_per_layer"]) == 2


class TestGradientSaturationAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_saturation_analysis(model, tokens)
        assert "layer_saturation" in result
        assert "max_saturation_layer" in result
        assert "pre_activation_means" in result
        assert "fraction_saturated" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_saturation_analysis(model, tokens)
        assert len(result["layer_saturation"]) == 2
        assert len(result["fraction_saturated"]) == 2

    def test_saturation_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = gradient_saturation_analysis(model, tokens)
        assert np.all(result["layer_saturation"] >= 0)
        assert np.all(result["layer_saturation"] <= 1.0 + 1e-5)


class TestLayernormGradientEffect:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layernorm_gradient_effect(model, tokens)
        assert "scale_factors" in result
        assert "input_norms" in result
        assert "output_norms" in result
        assert "compression_ratio" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = layernorm_gradient_effect(model, tokens)
        assert len(result["scale_factors"]) == 2
        assert len(result["input_norms"]) == 2


class TestPerHeadGradientSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_head_gradient_sensitivity(model, tokens, layer=0)
        assert "head_sensitivities" in result
        assert "most_sensitive_head" in result
        assert "least_sensitive_head" in result
        assert "sensitivity_ratio" in result

    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_head_gradient_sensitivity(model, tokens, layer=0)
        assert len(result["head_sensitivities"]) == 4

    def test_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_head_gradient_sensitivity(model, tokens, layer=0)
        assert np.all(result["head_sensitivities"] >= 0)
