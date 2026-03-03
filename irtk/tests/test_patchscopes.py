"""Tests for patchscopes representation inspection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.patchscopes import (
    patchscope,
    logit_lens_patchscope,
    token_identity_inspection,
    attribute_to_source,
    cross_model_inspection,
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


# ─── Patchscope Core ────────────────────────────────────────────────────────


class TestPatchscope:
    def test_returns_logits(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5, 6])
        result = patchscope(model, source, "blocks.0.hook_resid_post",
                            target, "blocks.1.hook_resid_post", target_pos=-1)
        assert result.shape == (3, 50)

    def test_patching_changes_output(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5, 6])
        clean = np.array(model(target))
        patched = patchscope(model, source, "blocks.0.hook_resid_post",
                             target, "blocks.1.hook_resid_post", target_pos=-1)
        assert not np.allclose(clean, patched, atol=1e-5)

    def test_source_pos(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        r1 = patchscope(model, source, "blocks.0.hook_resid_post",
                         target, "blocks.1.hook_resid_post", source_pos=0)
        r2 = patchscope(model, source, "blocks.0.hook_resid_post",
                         target, "blocks.1.hook_resid_post", source_pos=2)
        assert not np.allclose(r1, r2, atol=1e-5)

    def test_target_pos_explicit(self):
        model = _make_model()
        source = jnp.array([0, 1, 2])
        target = jnp.array([4, 5, 6])
        result = patchscope(model, source, "blocks.0.hook_resid_post",
                            target, "blocks.1.hook_resid_post", target_pos=0)
        assert result.shape == (3, 50)


# ─── Logit Lens Patchscope ──────────────────────────────────────────────────


class TestLogitLensPatchscope:
    def test_returns_logits(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens_patchscope(model, tokens, layer=0, pos=-1)
        assert result.shape == (50,)

    def test_different_layers_differ(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r0 = logit_lens_patchscope(model, tokens, layer=0, pos=-1)
        r1 = logit_lens_patchscope(model, tokens, layer=1, pos=-1)
        assert not np.allclose(r0, r1, atol=1e-5)

    def test_different_positions_differ(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r0 = logit_lens_patchscope(model, tokens, layer=0, pos=0)
        r1 = logit_lens_patchscope(model, tokens, layer=0, pos=2)
        assert not np.allclose(r0, r1, atol=1e-5)


# ─── Token Identity Inspection ──────────────────────────────────────────────


class TestTokenIdentityInspection:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_identity_inspection(model, tokens, layer=1, pos=-1, k=5)
        assert "top_tokens" in result
        assert "top_logits" in result
        assert "entropy" in result

    def test_top_k_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_identity_inspection(model, tokens, layer=0, pos=-1, k=5)
        assert len(result["top_tokens"]) == 5
        assert len(result["top_logits"]) == 5

    def test_entropy_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_identity_inspection(model, tokens, layer=0, pos=-1)
        assert result["entropy"] >= 0

    def test_probabilities_sum_roughly_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_identity_inspection(model, tokens, layer=0, pos=-1, k=50)
        total_prob = sum(p for _, p in result["top_tokens"])
        assert abs(total_prob - 1.0) < 0.01


# ─── Attribute to Source ─────────────────────────────────────────────────────


class TestAttributeToSource:
    def test_returns_dict(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        def metric(logits):
            return float(logits[-1, 0])
        result = attribute_to_source(model, source, target,
                                     "blocks.1.hook_resid_post", -1, metric,
                                     layers=[0])
        assert "attribution_matrix" in result
        assert "best_layer" in result
        assert "baseline_metric" in result

    def test_matrix_shape(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        def metric(logits):
            return float(logits[-1, 0])
        result = attribute_to_source(model, source, target,
                                     "blocks.1.hook_resid_post", -1, metric,
                                     layers=[0, 1])
        assert result["attribution_matrix"].shape == (2, 4)

    def test_baseline_metric_is_float(self):
        model = _make_model()
        source = jnp.array([0, 1, 2])
        target = jnp.array([4, 5])
        def metric(logits):
            return float(logits[-1, 0])
        result = attribute_to_source(model, source, target,
                                     "blocks.1.hook_resid_post", -1, metric)
        assert isinstance(result["baseline_metric"], float)


# ─── Cross-Model Inspection ─────────────────────────────────────────────────


class TestCrossModelInspection:
    def test_same_model_returns_logits(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5, 6])
        result = cross_model_inspection(
            model, model, source, "blocks.0.hook_resid_post",
            target, "blocks.1.hook_resid_post", target_pos=-1
        )
        assert result.shape == (3, 50)

    def test_different_models(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=99)
        source = jnp.array([0, 1, 2])
        target = jnp.array([4, 5])
        result = cross_model_inspection(
            model_a, model_b, source, "blocks.0.hook_resid_post",
            target, "blocks.1.hook_resid_post", target_pos=-1
        )
        assert result.shape == (2, 50)

    def test_with_projection(self):
        model = _make_model()
        source = jnp.array([0, 1, 2])
        target = jnp.array([4, 5])
        # Identity projection (same d_model)
        proj = jnp.eye(16)
        result = cross_model_inspection(
            model, model, source, "blocks.0.hook_resid_post",
            target, "blocks.1.hook_resid_post", target_pos=-1,
            projection=proj
        )
        assert result.shape == (2, 50)

    def test_source_pos(self):
        model = _make_model()
        source = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        r0 = cross_model_inspection(
            model, model, source, "blocks.0.hook_resid_post",
            target, "blocks.1.hook_resid_post", source_pos=0
        )
        r2 = cross_model_inspection(
            model, model, source, "blocks.0.hook_resid_post",
            target, "blocks.1.hook_resid_post", source_pos=2
        )
        assert not np.allclose(r0, r2, atol=1e-5)
