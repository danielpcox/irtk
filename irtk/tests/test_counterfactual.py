"""Tests for counterfactual and contrastive analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.counterfactual import (
    contrastive_activation_diff,
    minimal_change_tokens,
    counterfactual_effect_by_layer,
    token_necessity_sufficiency,
    contrastive_feature_attribution,
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


# ─── Contrastive Activation Diff ─────────────────────────────────────────


class TestContrastiveActivationDiff:
    def test_returns_dict(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_activation_diff(model, a, b)
        assert "layer_diffs" in result
        assert "divergence_layer" in result
        assert "max_diff_layer" in result
        assert "relative_diffs" in result

    def test_diffs_length(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_activation_diff(model, a, b)
        assert len(result["layer_diffs"]) == 3  # n_layers+1

    def test_diffs_non_negative(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_activation_diff(model, a, b)
        assert np.all(result["layer_diffs"] >= 0)

    def test_same_input_zero_diff(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        result = contrastive_activation_diff(model, a, a)
        assert np.all(result["layer_diffs"] < 1e-5)

    def test_max_diff_layer_valid(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_activation_diff(model, a, b)
        assert 0 <= result["max_diff_layer"] < 3


# ─── Minimal Change Tokens ───────────────────────────────────────────────


class TestMinimalChangeTokens:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = minimal_change_tokens(model, tokens, _metric, candidates=[10, 20])
        assert "best_pos" in result
        assert "best_replacement" in result
        assert "original_metric" in result
        assert "changed_metric" in result

    def test_best_pos_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = minimal_change_tokens(model, tokens, _metric, candidates=[10, 20])
        assert 0 <= result["best_pos"] < 4

    def test_per_position_effects_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = minimal_change_tokens(model, tokens, _metric, candidates=[10])
        assert len(result["per_position_effects"]) == 4

    def test_with_target_change(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = minimal_change_tokens(model, tokens, _metric, target_change=1.0, candidates=[10])
        assert "changed_metric" in result


# ─── Counterfactual Effect By Layer ───────────────────────────────────────


class TestCounterfactualEffectByLayer:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = counterfactual_effect_by_layer(model, clean, corrupt, _metric)
        assert "clean_metric" in result
        assert "corrupted_metric" in result
        assert "restoration_by_layer" in result
        assert "most_important_layer" in result

    def test_restoration_length(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = counterfactual_effect_by_layer(model, clean, corrupt, _metric)
        assert len(result["restoration_by_layer"]) == 2

    def test_most_important_valid(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = counterfactual_effect_by_layer(model, clean, corrupt, _metric)
        assert 0 <= result["most_important_layer"] < 2

    def test_metrics_are_floats(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = counterfactual_effect_by_layer(model, clean, corrupt, _metric)
        assert isinstance(result["clean_metric"], float)
        assert isinstance(result["corrupted_metric"], float)


# ─── Token Necessity Sufficiency ──────────────────────────────────────────


class TestTokenNecessitySufficiency:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_necessity_sufficiency(model, tokens, _metric)
        assert "necessity" in result
        assert "sufficiency" in result
        assert "most_necessary" in result
        assert "most_sufficient" in result

    def test_arrays_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_necessity_sufficiency(model, tokens, _metric)
        assert len(result["necessity"]) == 4
        assert len(result["sufficiency"]) == 4

    def test_most_necessary_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_necessity_sufficiency(model, tokens, _metric)
        assert 0 <= result["most_necessary"] < 4
        assert 0 <= result["most_sufficient"] < 4


# ─── Contrastive Feature Attribution ─────────────────────────────────────


class TestContrastiveFeatureAttribution:
    def test_returns_dict(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_feature_attribution(model, a, b, _metric)
        assert "attn_attribution" in result
        assert "mlp_attribution" in result
        assert "total_diff" in result
        assert "most_important_component" in result

    def test_attribution_length(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_feature_attribution(model, a, b, _metric)
        assert len(result["attn_attribution"]) == 2
        assert len(result["mlp_attribution"]) == 2

    def test_total_diff_consistent(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_feature_attribution(model, a, b, _metric)
        expected = _metric(model(a)) - _metric(model(b))
        assert abs(result["total_diff"] - expected) < 1e-4

    def test_most_important_tuple(self):
        model = _make_model()
        a = jnp.array([0, 1, 2, 3])
        b = jnp.array([4, 5, 6, 7])
        result = contrastive_feature_attribution(model, a, b, _metric)
        comp_type, layer = result["most_important_component"]
        assert comp_type in ("attn", "mlp")
        assert 0 <= layer < 2
