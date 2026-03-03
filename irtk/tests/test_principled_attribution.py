"""Tests for principled (Shapley-based) attribution methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.principled_attribution import (
    shapley_value_tokens,
    kernel_shap_activations,
    interaction_index,
    path_attribution_value,
    importance_ranking_with_std,
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


# ─── Shapley Value Tokens ──────────────────────────────────────────────────


class TestShapleyValueTokens:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = shapley_value_tokens(model, tokens, _metric, n_samples=5)
        assert "shapley_values" in result
        assert "null_value" in result
        assert "full_value" in result

    def test_shapley_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = shapley_value_tokens(model, tokens, _metric, n_samples=5)
        assert len(result["shapley_values"]) == 4

    def test_full_value_is_float(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = shapley_value_tokens(model, tokens, _metric, n_samples=3)
        assert isinstance(result["full_value"], float)

    def test_efficiency_gap_small(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = shapley_value_tokens(model, tokens, _metric, n_samples=50)
        # With enough samples, efficiency gap should be reasonable
        assert result["efficiency_gap"] < abs(result["full_value"] - result["null_value"]) + 1.0


# ─── Kernel SHAP Activations ──────────────────────────────────────────────


class TestKernelShapActivations:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kernel_shap_activations(model, tokens, "blocks.0.hook_resid_post",
                                          _metric, n_samples=20)
        assert "dimension_importance" in result
        assert "top_dimensions" in result
        assert "r_squared" in result

    def test_importance_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kernel_shap_activations(model, tokens, "blocks.0.hook_resid_post",
                                          _metric, n_samples=20)
        assert result["dimension_importance"].shape == (16,)

    def test_top_dimensions_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kernel_shap_activations(model, tokens, "blocks.0.hook_resid_post",
                                          _metric, n_samples=20)
        assert len(result["top_dimensions"]) <= 10

    def test_missing_hook(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = kernel_shap_activations(model, tokens, "nonexistent_hook",
                                          _metric, n_samples=5)
        assert len(result["top_dimensions"]) == 0


# ─── Interaction Index ─────────────────────────────────────────────────────


class TestInteractionIndex:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        comps = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"]
        result = interaction_index(model, tokens, _metric, comps)
        assert "interaction_matrix" in result
        assert "individual_effects" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        comps = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out",
                 "blocks.1.hook_attn_out"]
        result = interaction_index(model, tokens, _metric, comps)
        assert result["interaction_matrix"].shape == (3, 3)

    def test_individual_effects_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        comps = ["blocks.0.hook_attn_out", "blocks.1.hook_mlp_out"]
        result = interaction_index(model, tokens, _metric, comps)
        assert len(result["individual_effects"]) == 2

    def test_symmetric(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        comps = ["blocks.0.hook_attn_out", "blocks.1.hook_mlp_out"]
        result = interaction_index(model, tokens, _metric, comps)
        m = result["interaction_matrix"]
        assert abs(m[0, 1] - m[1, 0]) < 1e-6


# ─── Path Attribution Value ────────────────────────────────────────────────


class TestPathAttributionValue:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_attribution_value(model, tokens, _metric)
        assert "layer_attributions" in result
        assert "attn_attributions" in result
        assert "mlp_attributions" in result

    def test_attribution_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_attribution_value(model, tokens, _metric)
        assert len(result["layer_attributions"]) == 2
        assert len(result["attn_attributions"]) == 2
        assert len(result["mlp_attributions"]) == 2

    def test_layer_is_sum(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_attribution_value(model, tokens, _metric)
        for i in range(2):
            expected = result["attn_attributions"][i] + result["mlp_attributions"][i]
            assert abs(result["layer_attributions"][i] - expected) < 1e-6


# ─── Importance Ranking with Std ───────────────────────────────────────────


class TestImportanceRankingWithStd:
    def test_returns_dict(self):
        model = _make_model()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        comps = ["blocks.0.hook_attn_out", "blocks.1.hook_mlp_out"]
        result = importance_ranking_with_std(model, tokens_list, _metric, comps)
        assert "ranking" in result
        assert "effect_matrix" in result
        assert "significant_components" in result

    def test_ranking_length(self):
        model = _make_model()
        tokens_list = [jnp.array([0, 1, 2, 3])]
        comps = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"]
        result = importance_ranking_with_std(model, tokens_list, _metric, comps)
        assert len(result["ranking"]) == 2

    def test_effect_matrix_shape(self):
        model = _make_model()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        comps = ["blocks.0.hook_attn_out", "blocks.1.hook_mlp_out"]
        result = importance_ranking_with_std(model, tokens_list, _metric, comps)
        assert result["effect_matrix"].shape == (2, 2)

    def test_ranking_sorted_by_abs(self):
        model = _make_model()
        tokens_list = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        comps = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out",
                 "blocks.1.hook_attn_out", "blocks.1.hook_mlp_out"]
        result = importance_ranking_with_std(model, tokens_list, _metric, comps)
        abs_means = [abs(m) for _, m, _ in result["ranking"]]
        for i in range(len(abs_means) - 1):
            assert abs_means[i] >= abs_means[i + 1] - 1e-10
