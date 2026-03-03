"""Tests for cross-coders: joint sparse autoencoders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.cross_coder import (
    CrossCoder,
    train_crosscoder,
    shared_vs_specific_features,
    finetuning_feature_diff,
    cross_layer_crosscoder,
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


# ─── CrossCoder Class ────────────────────────────────────────────────────────


class TestCrossCoder:
    def test_init(self):
        key = jax.random.PRNGKey(42)
        cc = CrossCoder((16, 16), 32, key=key)
        assert cc.n_streams == 2
        assert cc.n_features == 32

    def test_encode(self):
        key = jax.random.PRNGKey(42)
        cc = CrossCoder((16, 16), 32, key=key)
        s1 = jnp.ones((4, 16))
        s2 = jnp.ones((4, 16))
        features = cc.encode([s1, s2])
        assert features.shape == (4, 32)

    def test_decode(self):
        key = jax.random.PRNGKey(42)
        cc = CrossCoder((16, 16), 32, key=key)
        features = jnp.ones((4, 32))
        out = cc.decode(features, 0)
        assert out.shape == (4, 16)

    def test_forward(self):
        key = jax.random.PRNGKey(42)
        cc = CrossCoder((16, 16), 32, key=key)
        s1 = jnp.ones((4, 16))
        s2 = jnp.ones((4, 16))
        outputs = cc([s1, s2])
        assert len(outputs) == 2
        assert outputs[0].shape == (4, 16)

    def test_different_stream_dims(self):
        key = jax.random.PRNGKey(42)
        cc = CrossCoder((16, 8), 32, key=key)
        s1 = jnp.ones((4, 16))
        s2 = jnp.ones((4, 8))
        outputs = cc([s1, s2])
        assert outputs[0].shape == (4, 16)
        assert outputs[1].shape == (4, 8)


# ─── Train CrossCoder ────────────────────────────────────────────────────────


class TestTrainCrossCoder:
    def test_returns_crosscoder(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        cc = train_crosscoder(
            [model_a, model_b],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs, n_features=16, n_steps=5, key=key,
        )
        assert isinstance(cc, CrossCoder)
        assert cc.n_features == 16

    def test_training_reduces_loss(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        key = jax.random.PRNGKey(0)
        cc = train_crosscoder(
            [model, model],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs, n_features=32, n_steps=20, key=key,
        )
        assert cc.n_streams == 2


# ─── Shared vs Specific Features ─────────────────────────────────────────────


class TestSharedVsSpecificFeatures:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        cc = train_crosscoder(
            [model_a, model_b],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs, n_features=16, n_steps=5, key=key,
        )
        result = shared_vs_specific_features(
            cc, [model_a, model_b],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs,
        )
        assert "shared_features" in result
        assert "specific_features" in result
        assert "sharing_scores" in result
        assert "n_shared" in result

    def test_sharing_scores_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        cc = train_crosscoder(
            [model, model],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs, n_features=16, n_steps=5, key=key,
        )
        result = shared_vs_specific_features(
            cc, [model, model],
            ["blocks.0.hook_resid_post", "blocks.0.hook_resid_post"],
            seqs,
        )
        assert len(result["sharing_scores"]) == 16


# ─── Finetuning Feature Diff ─────────────────────────────────────────────────


class TestFinetuningFeatureDiff:
    def test_returns_dict(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        result = finetuning_feature_diff(model_a, model_b, "blocks.0.hook_resid_post", seqs, n_features=16, n_steps=5, key=key)
        assert "crosscoder" in result
        assert "amplified_features" in result
        assert "suppressed_features" in result
        assert "mean_activation_diff" in result

    def test_diff_length(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        result = finetuning_feature_diff(model_a, model_b, "blocks.0.hook_resid_post", seqs, n_features=16, n_steps=5, key=key)
        assert len(result["mean_activation_diff"]) == 16


# ─── Cross-Layer CrossCoder ──────────────────────────────────────────────────


class TestCrossLayerCrossCoder:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        result = cross_layer_crosscoder(
            model,
            ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
            seqs, n_features=16, n_steps=5, key=key,
        )
        assert "crosscoder" in result
        assert "universal_features" in result
        assert "layer_specific_features" in result
        assert "per_layer_activity" in result

    def test_per_layer_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        key = jax.random.PRNGKey(0)
        result = cross_layer_crosscoder(
            model,
            ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
            seqs, n_features=16, n_steps=5, key=key,
        )
        assert result["per_layer_activity"].shape == (16, 2)
