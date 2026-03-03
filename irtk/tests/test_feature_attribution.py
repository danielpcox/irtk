"""Tests for feature-level attribution tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.feature_attribution import (
    token_to_neuron_attribution,
    token_to_direction_attribution,
    decompose_logit_by_token,
    feature_importance_ranking,
    cross_layer_attribution,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestTokenToNeuronAttribution:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_to_neuron_attribution(model, tokens, layer=0, neuron=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_to_neuron_attribution(model, tokens, layer=0, neuron=0)
        assert all(v >= 0 for v in result)

    def test_different_neurons_different_attributions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        a0 = token_to_neuron_attribution(model, tokens, layer=0, neuron=0)
        a1 = token_to_neuron_attribution(model, tokens, layer=0, neuron=1)
        # Different neurons should generally give different attributions
        assert not np.allclose(a0, a1, atol=1e-6)


class TestTokenToDirectionAttribution:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = token_to_direction_attribution(
            model, tokens, "blocks.0.hook_resid_post", direction
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = token_to_direction_attribution(
            model, tokens, "blocks.0.hook_resid_post", direction
        )
        assert all(v >= -1e-6 for v in result)

    def test_different_directions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        d1 = np.array([1.0] + [0.0] * 15, dtype=np.float32)
        d2 = np.array([0.0, 1.0] + [0.0] * 14, dtype=np.float32)
        a1 = token_to_direction_attribution(
            model, tokens, "blocks.0.hook_resid_post", d1
        )
        a2 = token_to_direction_attribution(
            model, tokens, "blocks.0.hook_resid_post", d2
        )
        # Different directions should give different attributions
        assert not np.allclose(a1, a2, atol=1e-6)


class TestDecomposeLogitByToken:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = decompose_logit_by_token(model, tokens, target_token=5)
        assert "attributions" in result
        assert "baseline_logit" in result
        assert "total_attribution" in result

    def test_attributions_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = decompose_logit_by_token(model, tokens, target_token=5)
        assert result["attributions"].shape == (4,)

    def test_baseline_logit_is_float(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = decompose_logit_by_token(model, tokens, target_token=5)
        assert isinstance(result["baseline_logit"], float)

    def test_different_targets(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r1 = decompose_logit_by_token(model, tokens, target_token=5)
        r2 = decompose_logit_by_token(model, tokens, target_token=10)
        assert r1["baseline_logit"] != r2["baseline_logit"]


class TestFeatureImportanceRanking:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_importance_ranking(
            model, tokens, "blocks.0.hook_resid_post", k=5
        )
        assert "top_indices" in result
        assert "top_values" in result
        assert "top_magnitudes" in result
        assert "full_activations" in result

    def test_top_k_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_importance_ranking(
            model, tokens, "blocks.0.hook_resid_post", k=5
        )
        assert len(result["top_indices"]) == 5
        assert len(result["top_values"]) == 5
        assert len(result["top_magnitudes"]) == 5

    def test_magnitudes_descending(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_importance_ranking(
            model, tokens, "blocks.0.hook_resid_post", k=10
        )
        mags = result["top_magnitudes"]
        for i in range(len(mags) - 1):
            assert mags[i] >= mags[i + 1] - 1e-6

    def test_full_activations_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_importance_ranking(
            model, tokens, "blocks.0.hook_resid_post", k=5
        )
        assert result["full_activations"].shape == (16,)

    def test_invalid_hook(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_importance_ranking(
            model, tokens, "nonexistent_hook", k=5
        )
        assert len(result["top_indices"]) == 0


class TestCrossLayerAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = cross_layer_attribution(
            model, tokens,
            "blocks.0.hook_resid_post",
            "blocks.1.hook_resid_post",
            direction, n_dims=5,
        )
        assert "source_dims" in result
        assert "attributions" in result
        assert "baseline_projection" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = cross_layer_attribution(
            model, tokens,
            "blocks.0.hook_resid_post",
            "blocks.1.hook_resid_post",
            direction, n_dims=5,
        )
        assert len(result["source_dims"]) == 5
        assert len(result["attributions"]) == 5

    def test_invalid_hooks(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = cross_layer_attribution(
            model, tokens,
            "nonexistent",
            "blocks.1.hook_resid_post",
            direction,
        )
        assert len(result["source_dims"]) == 0
