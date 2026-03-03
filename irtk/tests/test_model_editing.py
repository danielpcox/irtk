"""Tests for model editing tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.model_editing import (
    locate_decisive_layer,
    compute_key_vector,
    compute_value_target,
    apply_rank_one_edit,
    edit_fact,
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


# ─── Locate Decisive Layer ───────────────────────────────────────────────────


class TestLocateDecisiveLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = locate_decisive_layer(model, tokens, _metric, corrupt_pos=1)
        assert isinstance(result, dict)
        assert "decisive_layer" in result
        assert "layer_effects" in result

    def test_layer_effects_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = locate_decisive_layer(model, tokens, _metric, corrupt_pos=1)
        assert result["layer_effects"].shape == (2,)

    def test_decisive_layer_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = locate_decisive_layer(model, tokens, _metric, corrupt_pos=1)
        assert 0 <= result["decisive_layer"] < 2

    def test_clean_metric_matches(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = locate_decisive_layer(model, tokens, _metric, corrupt_pos=1)
        expected = _metric(model(tokens))
        assert abs(result["clean_metric"] - expected) < 1e-4


# ─── Compute Key Vector ─────────────────────────────────────────────────────


class TestComputeKeyVector:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compute_key_vector(model, tokens, layer=0, pos=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16,)

    def test_nonzero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compute_key_vector(model, tokens, layer=0, pos=1)
        assert np.linalg.norm(result) > 0

    def test_different_positions_different(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        k0 = compute_key_vector(model, tokens, layer=0, pos=0)
        k1 = compute_key_vector(model, tokens, layer=0, pos=1)
        assert not np.allclose(k0, k1, atol=1e-6)

    def test_different_layers_different(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        k0 = compute_key_vector(model, tokens, layer=0, pos=1)
        k1 = compute_key_vector(model, tokens, layer=1, pos=1)
        assert not np.allclose(k0, k1, atol=1e-6)


# ─── Compute Value Target ───────────────────────────────────────────────────


class TestComputeValueTarget:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compute_value_target(model, tokens, layer=0, pos=-1, target_token=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16,)

    def test_nonzero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = compute_value_target(model, tokens, layer=0, pos=-1, target_token=5)
        assert np.linalg.norm(result) > 0

    def test_different_targets_different(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        v1 = compute_value_target(model, tokens, layer=0, pos=-1, target_token=5)
        v2 = compute_value_target(model, tokens, layer=0, pos=-1, target_token=10)
        assert not np.allclose(v1, v2, atol=1e-6)

    def test_coeff_scales(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        v1 = compute_value_target(model, tokens, layer=0, pos=-1, target_token=5, coeff=1.0)
        v2 = compute_value_target(model, tokens, layer=0, pos=-1, target_token=5, coeff=2.0)
        np.testing.assert_allclose(v2, v1 * 2.0, atol=1e-5)


# ─── Apply Rank-One Edit ────────────────────────────────────────────────────


class TestApplyRankOneEdit:
    def test_returns_model(self):
        model = _make_model()
        key_vec = np.ones(16, dtype=np.float32) * 0.1
        val_vec = np.ones(16, dtype=np.float32) * 0.1
        result = apply_rank_one_edit(model, layer=0, key_vector=key_vec, value_vector=val_vec)
        assert isinstance(result, HookedTransformer)

    def test_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        key_vec = np.random.randn(16).astype(np.float32)
        val_vec = np.random.randn(16).astype(np.float32)
        edited = apply_rank_one_edit(model, layer=0, key_vector=key_vec, value_vector=val_vec)
        logits_after = edited(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_only_edits_target_layer(self):
        model = _make_model()
        key_vec = np.ones(16, dtype=np.float32) * 0.1
        val_vec = np.ones(16, dtype=np.float32) * 0.1
        edited = apply_rank_one_edit(model, layer=0, key_vector=key_vec, value_vector=val_vec)
        # Layer 1 should be unchanged
        assert np.allclose(edited.blocks[1].mlp.W_out, model.blocks[1].mlp.W_out)

    def test_original_unchanged(self):
        model = _make_model()
        W_out_orig = np.array(model.blocks[0].mlp.W_out)
        key_vec = np.ones(16, dtype=np.float32)
        val_vec = np.ones(16, dtype=np.float32)
        _ = apply_rank_one_edit(model, layer=0, key_vector=key_vec, value_vector=val_vec)
        assert np.allclose(model.blocks[0].mlp.W_out, W_out_orig)


# ─── Edit Fact (High-Level API) ──────────────────────────────────────────────


class TestEditFact:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edit_fact(model, tokens, subject_pos=1, target_token=5, layer=0)
        assert isinstance(result, dict)
        assert "edited_model" in result
        assert "layer" in result

    def test_edited_model_is_model(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edit_fact(model, tokens, subject_pos=1, target_token=5, layer=0)
        assert isinstance(result["edited_model"], HookedTransformer)

    def test_auto_detect_layer(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edit_fact(model, tokens, subject_pos=1, target_token=5)
        assert 0 <= result["layer"] < 2

    def test_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        result = edit_fact(model, tokens, subject_pos=1, target_token=5, layer=0, coeff=5.0)
        logits_after = result["edited_model"](tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_has_key_and_value_vectors(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edit_fact(model, tokens, subject_pos=1, target_token=5, layer=0)
        assert result["key_vector"].shape == (16,)
        assert result["value_vector"].shape == (16,)
