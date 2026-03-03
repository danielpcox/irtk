"""Tests for in-context learning mechanism analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.icl_mechanisms import (
    extract_task_vector,
    icl_head_identification,
    implicit_gradient_descent_test,
    icl_label_sensitivity,
    demonstration_order_effect,
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


# ─── Extract Task Vector ─────────────────────────────────────────────────────


class TestExtractTaskVector:
    def test_returns_dict(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = extract_task_vector(model, demo, query, "blocks.0.hook_resid_post")
        assert "task_vector" in result
        assert "task_vector_norm" in result
        assert "baseline_norm" in result

    def test_task_vector_shape(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = extract_task_vector(model, demo, query, "blocks.0.hook_resid_post")
        assert result["task_vector"].shape == (16,)

    def test_same_input_zero_vector(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_task_vector(model, tokens, tokens, "blocks.0.hook_resid_post")
        np.testing.assert_allclose(result["task_vector_norm"], 0.0, atol=1e-5)

    def test_norm_non_negative(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = extract_task_vector(model, demo, query, "blocks.0.hook_resid_post")
        assert result["task_vector_norm"] >= 0


# ─── ICL Head Identification ─────────────────────────────────────────────────


class TestICLHeadIdentification:
    def test_returns_dict(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = icl_head_identification(model, demo, query, _metric)
        assert "head_icl_scores" in result
        assert "top_icl_heads" in result
        assert "total_icl_effect" in result

    def test_scores_shape(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = icl_head_identification(model, demo, query, _metric)
        assert result["head_icl_scores"].shape == (2, 4)

    def test_scores_non_negative(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = icl_head_identification(model, demo, query, _metric)
        assert np.all(result["head_icl_scores"] >= 0)

    def test_top_heads_sorted(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = icl_head_identification(model, demo, query, _metric)
        scores = [s for _, _, s in result["top_icl_heads"]]
        assert scores == sorted(scores, reverse=True)


# ─── Implicit Gradient Descent Test ──────────────────────────────────────────


class TestImplicitGradientDescentTest:
    def test_returns_dict(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = implicit_gradient_descent_test(model, demo, query, layer=0)
        assert "alignment_score" in result
        assert "attention_update_norm" in result
        assert "layer_tested" in result

    def test_alignment_in_range(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = implicit_gradient_descent_test(model, demo, query, layer=0)
        assert -1.0 <= result["alignment_score"] <= 1.0

    def test_layer_recorded(self):
        model = _make_model()
        demo = jnp.array([0, 1, 2, 3, 4, 5])
        query = jnp.array([4, 5])
        result = implicit_gradient_descent_test(model, demo, query, layer=1)
        assert result["layer_tested"] == 1


# ─── ICL Label Sensitivity ───────────────────────────────────────────────────


class TestICLLabelSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = icl_label_sensitivity(model, tokens, _metric, corruption_positions=[2, 4])
        assert "clean_metric" in result
        assert "mean_corrupted_metric" in result
        assert "sensitivity_score" in result

    def test_sensitivity_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = icl_label_sensitivity(model, tokens, _metric, corruption_positions=[2, 4])
        assert result["sensitivity_score"] >= 0

    def test_corrupted_metrics_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = icl_label_sensitivity(model, tokens, _metric, corruption_positions=[2], n_corruptions=3)
        assert len(result["corrupted_metrics"]) == 3


# ─── Demonstration Order Effect ──────────────────────────────────────────────


class TestDemonstrationOrderEffect:
    def test_returns_dict(self):
        model = _make_model()
        demos = [jnp.array([0, 1]), jnp.array([2, 3]), jnp.array([4, 5])]
        query = jnp.array([6, 7])
        result = demonstration_order_effect(model, demos, query, _metric, n_shuffles=5)
        assert "mean_metric" in result
        assert "std_metric" in result
        assert "order_sensitivity" in result

    def test_std_non_negative(self):
        model = _make_model()
        demos = [jnp.array([0, 1]), jnp.array([2, 3])]
        query = jnp.array([6, 7])
        result = demonstration_order_effect(model, demos, query, _metric, n_shuffles=5)
        assert result["std_metric"] >= 0

    def test_min_max_consistent(self):
        model = _make_model()
        demos = [jnp.array([0, 1]), jnp.array([2, 3]), jnp.array([4, 5])]
        query = jnp.array([6, 7])
        result = demonstration_order_effect(model, demos, query, _metric, n_shuffles=5)
        assert result["min_metric"] <= result["max_metric"]
