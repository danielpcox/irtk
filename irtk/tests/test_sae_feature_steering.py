"""Tests for SAE feature-level steering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.sae_feature_steering import (
    feature_steer,
    multi_feature_steer,
    find_steering_features,
    feature_ablation_effect,
    clamped_feature_generation,
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


def _make_sae(d_model=16, n_features=32, seed=0):
    key = jax.random.PRNGKey(seed)
    return SparseAutoencoder(d_model, n_features, key=key)


def _metric(logits):
    return float(logits[-1, 0])


# ─── Feature Steer ───────────────────────────────────────────────────────────


class TestFeatureSteer:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_steer(model, tokens, sae, "blocks.0.hook_resid_post", 0)
        assert "steered_logits" in result
        assert "clean_logits" in result
        assert "feature_activation" in result
        assert "logit_diff" in result

    def test_alpha_zero_is_ablation(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_steer(model, tokens, sae, "blocks.0.hook_resid_post", 0, alpha=0.0)
        assert result["steered_logits"].shape == result["clean_logits"].shape

    def test_alpha_one_is_identity(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_steer(model, tokens, sae, "blocks.0.hook_resid_post", 0, alpha=1.0)
        # Alpha=1 means no change (multiply by 1)
        np.testing.assert_allclose(
            result["steered_logits"], result["clean_logits"], atol=1e-3
        )

    def test_pos_argument(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_steer(model, tokens, sae, "blocks.0.hook_resid_post", 0, alpha=5.0, pos=2)
        assert "logit_diff" in result


# ─── Multi Feature Steer ────────────────────────────────────────────────────


class TestMultiFeatureSteer:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_feature_steer(
            model, tokens, sae, "blocks.0.hook_resid_post", {0: 2.0, 1: 0.0}
        )
        assert "steered_logits" in result
        assert result["n_features_modified"] == 2

    def test_empty_interventions(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_feature_steer(
            model, tokens, sae, "blocks.0.hook_resid_post", {}
        )
        assert result["n_features_modified"] == 0
        np.testing.assert_allclose(
            result["steered_logits"], result["clean_logits"], atol=1e-5
        )


# ─── Find Steering Features ─────────────────────────────────────────────────


class TestFindSteeringFeatures:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = find_steering_features(
            model, tokens, sae, "blocks.0.hook_resid_post", _metric
        )
        assert "feature_effects" in result
        assert "baseline_metric" in result

    def test_top_k_limit(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = find_steering_features(
            model, tokens, sae, "blocks.0.hook_resid_post", _metric, top_k=3
        )
        assert len(result["feature_effects"]) <= 3


# ─── Feature Ablation Effect ────────────────────────────────────────────────


class TestFeatureAblationEffect:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_ablation_effect(
            model, tokens, sae, "blocks.0.hook_resid_post", _metric
        )
        assert "ablation_effects" in result
        assert "baseline_metric" in result

    def test_specific_features(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_ablation_effect(
            model, tokens, sae, "blocks.0.hook_resid_post", _metric,
            features=[0, 1, 2]
        )
        assert len(result["ablation_effects"]) == 3

    def test_total_effect_non_negative(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_ablation_effect(
            model, tokens, sae, "blocks.0.hook_resid_post", _metric,
            features=[0, 1]
        )
        assert result["total_effect"] >= 0


# ─── Clamped Feature Generation ─────────────────────────────────────────────


class TestClampedFeatureGeneration:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = clamped_feature_generation(
            model, tokens, sae, "blocks.0.hook_resid_post", 0,
            clamp_value=1.0, max_new_tokens=3,
        )
        assert "generated_tokens" in result
        assert result["n_generated"] == 3
        assert result["clamped_feature"] == 0
        assert len(result["generated_tokens"]) == 7  # 4 + 3

    def test_greedy_generation(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2])
        result = clamped_feature_generation(
            model, tokens, sae, "blocks.0.hook_resid_post", 0,
            clamp_value=2.0, max_new_tokens=2, temperature=0.0,
        )
        assert len(result["generated_tokens"]) == 5
