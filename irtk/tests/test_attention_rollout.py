"""Tests for attention rollout and flow tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.attention_rollout import (
    layer_aggregated_attention,
    effective_attention,
    attention_rollout,
    attention_flow,
    token_attribution_rollout,
    per_head_rollout,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _get_cache(model, tokens):
    _, cache = model.run_with_cache(tokens)
    return cache


# ─── Layer Aggregated Attention ──────────────────────────────────────────────


class TestLayerAggregatedAttention:
    def test_returns_matrix(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = layer_aggregated_attention(cache, layer=0, n_heads=4)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)

    def test_rows_sum_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = layer_aggregated_attention(cache, layer=0, n_heads=4, method="mean")
        # Mean of softmax rows should sum to ~1
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = layer_aggregated_attention(cache, layer=0, n_heads=4)
        assert np.all(result >= -1e-7)

    def test_max_method(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        mean_result = layer_aggregated_attention(cache, layer=0, n_heads=4, method="mean")
        max_result = layer_aggregated_attention(cache, layer=0, n_heads=4, method="max")
        # Max should be >= mean element-wise
        assert np.all(max_result >= mean_result - 1e-7)


# ─── Effective Attention ─────────────────────────────────────────────────────


class TestEffectiveAttention:
    def test_returns_matrix(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = effective_attention(cache, layer=0, n_heads=4)
        assert result.shape == (4, 4)

    def test_with_zero_weight_is_identity(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = effective_attention(cache, layer=0, n_heads=4, residual_weight=0.0)
        np.testing.assert_allclose(result, np.eye(4), atol=1e-6)

    def test_with_full_weight_is_attention(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn = layer_aggregated_attention(cache, layer=0, n_heads=4)
        result = effective_attention(cache, layer=0, n_heads=4, residual_weight=1.0)
        np.testing.assert_allclose(result, attn, atol=1e-6)

    def test_rows_sum_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        result = effective_attention(cache, layer=0, n_heads=4, residual_weight=0.5)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ─── Attention Rollout ───────────────────────────────────────────────────────


class TestAttentionRollout:
    def test_returns_matrix(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_rollout(model, tokens)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)

    def test_rows_sum_approximately_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_rollout(model, tokens)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.1)

    def test_single_layer(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn_l0 = layer_aggregated_attention(cache, layer=0, n_heads=4)
        # Rollout of just layer 0 should be the attention pattern at layer 0
        result = attention_rollout(model, tokens, start_layer=0, end_layer=1)
        np.testing.assert_allclose(result, attn_l0, atol=1e-5)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_rollout(model, tokens)
        assert np.all(result >= -1e-6)


# ─── Attention Flow ──────────────────────────────────────────────────────────


class TestAttentionFlow:
    def test_returns_matrix(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_flow(model, tokens)
        assert result.shape == (4, 4)

    def test_with_zero_residual_is_identity(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_flow(model, tokens, residual_weight=0.0)
        np.testing.assert_allclose(result, np.eye(4), atol=1e-6)

    def test_with_full_residual_matches_rollout(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        flow = attention_flow(model, tokens, residual_weight=1.0)
        rollout = attention_rollout(model, tokens)
        np.testing.assert_allclose(flow, rollout, atol=1e-5)

    def test_rows_sum_approximately_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_flow(model, tokens, residual_weight=0.5)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.1)


# ─── Token Attribution Rollout ───────────────────────────────────────────────


class TestTokenAttributionRollout:
    def test_returns_vector(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_attribution_rollout(model, tokens)
        assert result.shape == (4,)

    def test_sums_approximately_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_attribution_rollout(model, tokens)
        assert abs(result.sum() - 1.0) < 0.1

    def test_flow_method(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_attribution_rollout(model, tokens, method="flow", residual_weight=0.5)
        assert result.shape == (4,)

    def test_different_positions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r_last = token_attribution_rollout(model, tokens, output_pos=-1)
        r_first = token_attribution_rollout(model, tokens, output_pos=0)
        # Both should be valid attribution vectors with correct shape
        assert r_last.shape == (4,)
        assert r_first.shape == (4,)


# ─── Per Head Rollout ────────────────────────────────────────────────────────


class TestPerHeadRollout:
    def test_returns_vector(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_head_rollout(model, tokens, layer=0, head=0)
        assert result.shape == (4,)

    def test_different_heads_different_results(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r0 = per_head_rollout(model, tokens, layer=0, head=0)
        r1 = per_head_rollout(model, tokens, layer=0, head=1)
        assert not np.allclose(r0, r1, atol=1e-6)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_head_rollout(model, tokens, layer=0, head=0)
        assert np.all(result >= -1e-6)
