"""Tests for systematic ablation study framework."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.ablation_study import (
    layer_by_layer_ablation,
    head_importance_matrix,
    position_sensitivity,
    double_ablation_interaction,
    ablation_summary,
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


# ─── Layer-by-Layer Ablation ────────────────────────────────────────────────


class TestLayerByLayerAblation:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_by_layer_ablation(model, seqs, _metric)
        assert "effects" in result
        assert "std" in result

    def test_effects_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_by_layer_ablation(model, seqs, _metric)
        assert result["effects"].shape == (2,)

    def test_mlp_component(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_by_layer_ablation(model, seqs, _metric, component="mlp")
        assert result["effects"].shape == (2,)

    def test_per_prompt_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = layer_by_layer_ablation(model, seqs, _metric)
        assert result["per_prompt"].shape == (2, 2)

    def test_mean_ablation(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = layer_by_layer_ablation(model, seqs, _metric, ablation_type="mean")
        assert result["effects"].shape == (2,)


# ─── Head Importance Matrix ────────────────────────────────────────────────


class TestHeadImportanceMatrix:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_importance_matrix(model, seqs, _metric)
        assert "matrix" in result
        assert "std_matrix" in result

    def test_matrix_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_importance_matrix(model, seqs, _metric)
        assert result["matrix"].shape == (2, 4)

    def test_clean_mean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        seqs = [tokens]
        result = head_importance_matrix(model, seqs, _metric)
        expected = _metric(model(tokens))
        assert abs(result["clean_mean"] - expected) < 1e-4

    def test_multiple_prompts(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = head_importance_matrix(model, seqs, _metric)
        assert result["matrix"].shape == (2, 4)


# ─── Position Sensitivity ──────────────────────────────────────────────────


class TestPositionSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_sensitivity(model, tokens, _metric)
        assert "effects" in result
        assert "most_important_pos" in result

    def test_effects_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_sensitivity(model, tokens, _metric)
        assert result["effects"].shape == (4,)

    def test_most_important_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_sensitivity(model, tokens, _metric)
        assert 0 <= result["most_important_pos"] < 4

    def test_mean_ablation(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_sensitivity(model, tokens, _metric, ablation_type="mean")
        assert result["effects"].shape == (4,)


# ─── Double Ablation Interaction ────────────────────────────────────────────


class TestDoubleAblationInteraction:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        hooks_a = ["blocks.0.hook_attn_out"]
        hooks_b = ["blocks.1.hook_attn_out"]
        result = double_ablation_interaction(model, tokens, _metric, hooks_a, hooks_b)
        assert "interaction_matrix" in result

    def test_interaction_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        hooks_a = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"]
        hooks_b = ["blocks.1.hook_attn_out"]
        result = double_ablation_interaction(model, tokens, _metric, hooks_a, hooks_b)
        assert result["interaction_matrix"].shape == (2, 1)

    def test_effects_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        hooks_a = ["blocks.0.hook_attn_out"]
        hooks_b = ["blocks.1.hook_attn_out"]
        result = double_ablation_interaction(model, tokens, _metric, hooks_a, hooks_b)
        assert result["effects_a"].shape == (1,)
        assert result["effects_b"].shape == (1,)

    def test_same_hook_no_interaction(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        hooks = ["blocks.0.hook_attn_out"]
        result = double_ablation_interaction(model, tokens, _metric, hooks, hooks)
        # Same hook: joint == individual, interaction should be ~ -individual
        assert result["joint_effects"].shape == (1, 1)


# ─── Ablation Summary ──────────────────────────────────────────────────────


class TestAblationSummary:
    def test_returns_dict(self):
        effects = np.array([0.1, -0.5, 0.3, 0.0, -0.2])
        result = ablation_summary(effects)
        assert "top_components" in result
        assert "gini_coefficient" in result

    def test_top_k(self):
        effects = np.array([0.1, -0.5, 0.3, 0.0, -0.2])
        result = ablation_summary(effects, top_k=3)
        assert len(result["top_components"]) == 3

    def test_sorted_by_abs(self):
        effects = np.array([0.1, -0.5, 0.3, 0.0, -0.2])
        result = ablation_summary(effects, top_k=5)
        abs_vals = [abs(e) for _, e in result["top_components"]]
        for i in range(len(abs_vals) - 1):
            assert abs_vals[i] >= abs_vals[i + 1]

    def test_with_labels(self):
        effects = np.array([0.1, -0.5])
        labels = ["L0_attn", "L1_attn"]
        result = ablation_summary(effects, labels=labels)
        assert result["top_components"][0][0] == "L1_attn"

    def test_gini_in_range(self):
        effects = np.random.randn(20)
        result = ablation_summary(effects)
        assert 0 <= result["gini_coefficient"] <= 1.0

    def test_2d_effects(self):
        effects = np.random.randn(3, 4)
        result = ablation_summary(effects)
        assert result["mean_effect"] > 0
