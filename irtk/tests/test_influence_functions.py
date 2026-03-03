"""Tests for training data attribution via influence functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.influence_functions import (
    compute_influence_scores,
    influence_ablation_curve,
    training_example_attribution,
    influence_to_feature,
    counterfactual_training_effect,
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


# ─── Compute Influence Scores ────────────────────────────────────────────


class TestComputeInfluenceScores:
    def test_returns_dict(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = compute_influence_scores(model, test, train, target_token=0)
        assert "influence_scores" in result
        assert "top_influential" in result

    def test_scores_length(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = compute_influence_scores(model, test, train, target_token=0)
        assert len(result["influence_scores"]) == 2

    def test_top_influential_sorted(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = compute_influence_scores(model, test, train, target_token=0)
        abs_scores = [abs(s) for _, s in result["top_influential"]]
        assert abs_scores == sorted(abs_scores, reverse=True)


# ─── Influence Ablation Curve ─────────────────────────────────────────────


class TestInfluenceAblationCurve:
    def test_returns_dict(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = influence_ablation_curve(model, test, train, _metric, steps=3)
        assert "n_removed" in result
        assert "metrics" in result
        assert "baseline_metric" in result

    def test_lengths_match(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = influence_ablation_curve(model, test, train, _metric, steps=3)
        assert len(result["n_removed"]) == len(result["metrics"])

    def test_baseline_first(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7])]
        result = influence_ablation_curve(model, test, train, _metric, steps=2)
        assert result["n_removed"][0] == 0


# ─── Training Example Attribution ─────────────────────────────────────────


class TestTrainingExampleAttribution:
    def test_returns_dict(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7])]
        result = training_example_attribution(model, test, train, "blocks.0.hook_resid_post", 0)
        assert "attributions" in result
        assert "top_examples" in result
        assert "test_activation" in result

    def test_attributions_length(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = training_example_attribution(model, test, train, "blocks.0.hook_resid_post", 0)
        assert len(result["attributions"]) == 2


# ─── Influence to Feature ─────────────────────────────────────────────────


class TestInfluenceToFeature:
    def test_returns_dict(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7])]
        result = influence_to_feature(model, test, train, "blocks.0.hook_resid_post")
        assert "feature_influences" in result
        assert "most_influenced_feature" in result
        assert "per_feature_total" in result

    def test_influences_shape(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = influence_to_feature(model, test, train, "blocks.0.hook_resid_post")
        assert result["feature_influences"].shape == (2, 16)

    def test_per_feature_length(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = [jnp.array([4, 5, 6, 7])]
        result = influence_to_feature(model, test, train, "blocks.0.hook_resid_post")
        assert len(result["per_feature_total"]) == 16


# ─── Counterfactual Training Effect ──────────────────────────────────────


class TestCounterfactualTrainingEffect:
    def test_returns_dict(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = jnp.array([4, 5, 6, 7])
        result = counterfactual_training_effect(model, test, train, _metric)
        assert "original_metric" in result
        assert "counterfactual_metric" in result
        assert "effect" in result
        assert "alignment" in result

    def test_effect_is_float(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = jnp.array([4, 5, 6, 7])
        result = counterfactual_training_effect(model, test, train, _metric)
        assert isinstance(result["effect"], float)

    def test_alignment_in_range(self):
        model = _make_model()
        test = jnp.array([0, 1, 2, 3])
        train = jnp.array([4, 5, 6, 7])
        result = counterfactual_training_effect(model, test, train, _metric)
        assert -1.0 <= result["alignment"] <= 1.0
