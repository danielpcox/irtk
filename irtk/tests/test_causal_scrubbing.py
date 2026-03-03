"""Tests for causal scrubbing and intervention tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.causal_scrubbing import (
    causal_scrub,
    interchange_intervention,
    path_patching_matrix,
    corrupt_and_restore,
    multi_hook_scrub,
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


# ─── Causal Scrub ───────────────────────────────────────────────────────────


class TestCausalScrub:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, ["blocks.0.hook_attn_out"], _metric)
        assert isinstance(result, dict)
        assert "clean_metric" in result
        assert "scrubbed_metric" in result

    def test_clean_metric_matches(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, ["blocks.0.hook_attn_out"], _metric)
        expected = _metric(model(clean))
        assert abs(result["clean_metric"] - expected) < 1e-4

    def test_metric_change_sign(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, ["blocks.0.hook_attn_out"], _metric)
        # metric_change should equal scrubbed - clean
        expected_change = result["scrubbed_metric"] - result["clean_metric"]
        assert abs(result["metric_change"] - expected_change) < 1e-6

    def test_relative_change_nonneg(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, ["blocks.0.hook_attn_out"], _metric)
        assert result["relative_change"] >= 0

    def test_empty_hooks_no_change(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, [], _metric)
        assert abs(result["metric_change"]) < 1e-5

    def test_invalid_hook_ignored(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = causal_scrub(model, clean, ref, ["nonexistent_hook"], _metric)
        assert abs(result["metric_change"]) < 1e-5


# ─── Interchange Intervention ───────────────────────────────────────────────


class TestInterchangeIntervention:
    def test_returns_dict(self):
        model = _make_model()
        base = jnp.array([0, 1, 2, 3])
        source = jnp.array([4, 5, 6, 7])
        result = interchange_intervention(
            model, base, source, "blocks.0.hook_attn_out", _metric
        )
        assert isinstance(result, dict)
        assert "base_metric" in result
        assert "intervened_metric" in result

    def test_base_metric_matches(self):
        model = _make_model()
        base = jnp.array([0, 1, 2, 3])
        source = jnp.array([4, 5, 6, 7])
        result = interchange_intervention(
            model, base, source, "blocks.0.hook_attn_out", _metric
        )
        expected = _metric(model(base))
        assert abs(result["base_metric"] - expected) < 1e-4

    def test_metric_change_consistent(self):
        model = _make_model()
        base = jnp.array([0, 1, 2, 3])
        source = jnp.array([4, 5, 6, 7])
        result = interchange_intervention(
            model, base, source, "blocks.0.hook_attn_out", _metric
        )
        expected_change = result["intervened_metric"] - result["base_metric"]
        assert abs(result["metric_change"] - expected_change) < 1e-6

    def test_with_positions(self):
        model = _make_model()
        base = jnp.array([0, 1, 2, 3])
        source = jnp.array([4, 5, 6, 7])
        result = interchange_intervention(
            model, base, source, "blocks.0.hook_attn_out", _metric, positions=[1, 2]
        )
        assert "intervened_metric" in result

    def test_invalid_hook_returns_base(self):
        model = _make_model()
        base = jnp.array([0, 1, 2, 3])
        source = jnp.array([4, 5, 6, 7])
        result = interchange_intervention(
            model, base, source, "nonexistent_hook", _metric
        )
        assert result["metric_change"] == 0.0


# ─── Path Patching Matrix ──────────────────────────────────────────────────


class TestPathPatchingMatrix:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = path_patching_matrix(model, clean, corrupt, _metric)
        assert isinstance(result, dict)
        assert "matrix" in result

    def test_matrix_shape(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = path_patching_matrix(model, clean, corrupt, _metric)
        assert result["matrix"].shape == (2, 2)

    def test_layer_effects_shape(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = path_patching_matrix(model, clean, corrupt, _metric)
        assert result["layer_effects"].shape == (2,)

    def test_clean_metric_matches(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = path_patching_matrix(model, clean, corrupt, _metric)
        expected = _metric(model(clean))
        assert abs(result["clean_metric"] - expected) < 1e-4

    def test_diagonal_zero(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupt = jnp.array([4, 5, 6, 7])
        result = path_patching_matrix(model, clean, corrupt, _metric)
        # Diagonal should be zero (no self-patching)
        for i in range(2):
            assert abs(result["matrix"][i, i]) < 1e-6


# ─── Corrupt and Restore ───────────────────────────────────────────────────


class TestCorruptAndRestore:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = corrupt_and_restore(model, tokens, "blocks.0.hook_attn_out", _metric)
        assert isinstance(result, dict)
        assert "clean_metric" in result
        assert "corrupted_metric" in result

    def test_restored_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = corrupt_and_restore(model, tokens, "blocks.0.hook_attn_out", _metric)
        assert result["restored_at_layer"].shape == (2,)
        assert result["recovery_at_layer"].shape == (2,)

    def test_recovery_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = corrupt_and_restore(model, tokens, "blocks.0.hook_attn_out", _metric)
        expected = result["restored_at_layer"] - result["corrupted_metric"]
        np.testing.assert_allclose(result["recovery_at_layer"], expected, atol=1e-5)

    def test_custom_corrupt_fn(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])

        def noise_corrupt(x, name):
            return x + 0.5

        result = corrupt_and_restore(
            model, tokens, "blocks.0.hook_attn_out", _metric, corrupt_fn=noise_corrupt
        )
        assert "corrupted_metric" in result

    def test_clean_metric_matches(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = corrupt_and_restore(model, tokens, "blocks.0.hook_attn_out", _metric)
        expected = _metric(model(tokens))
        assert abs(result["clean_metric"] - expected) < 1e-4


# ─── Multi-Hook Scrub ──────────────────────────────────────────────────────


class TestMultiHookScrub:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        hooks = ["blocks.0.hook_attn_out", "blocks.1.hook_attn_out"]
        result = multi_hook_scrub(model, clean, ref, hooks, _metric)
        assert isinstance(result, dict)
        assert "per_hook_effects" in result

    def test_per_hook_effects_has_all_hooks(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        hooks = ["blocks.0.hook_attn_out", "blocks.1.hook_attn_out"]
        result = multi_hook_scrub(model, clean, ref, hooks, _metric)
        for hook in hooks:
            assert hook in result["per_hook_effects"]

    def test_most_important_in_hooks(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        hooks = ["blocks.0.hook_attn_out", "blocks.1.hook_attn_out"]
        result = multi_hook_scrub(model, clean, ref, hooks, _metric)
        assert result["most_important"] in hooks

    def test_clean_metric_matches(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        hooks = ["blocks.0.hook_attn_out"]
        result = multi_hook_scrub(model, clean, ref, hooks, _metric)
        expected = _metric(model(clean))
        assert abs(result["clean_metric"] - expected) < 1e-4

    def test_invalid_hooks_zero_effect(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        ref = jnp.array([4, 5, 6, 7])
        result = multi_hook_scrub(model, clean, ref, ["nonexistent"], _metric)
        assert result["per_hook_effects"]["nonexistent"] == 0.0
