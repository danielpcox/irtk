"""Tests for attention pattern analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.attention_utils import (
    entropy,
    all_head_entropy,
    head_pattern_similarity,
    attention_to_token,
    attention_from_token,
    top_attended_tokens,
    causal_tracing,
    max_attention_position,
    attention_head_summary,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


def _get_cache(model, tokens):
    _, cache = model.run_with_cache(tokens)
    return cache


class TestEntropy:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        ent = entropy(cache, layer=0, head=0)
        assert ent.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        ent = entropy(cache, layer=0, head=0)
        assert np.all(ent >= 0)

    def test_maximum_entropy(self):
        """Entropy should be at most log(seq_len) for uniform attention."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        ent = entropy(cache, layer=0, head=0)
        max_possible = np.log(4)  # log(seq_len)
        assert np.all(ent <= max_possible + 1e-5)


class TestAllHeadEntropy:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        ent = all_head_entropy(cache, model)
        assert ent.shape == (2, 4)  # n_layers, n_heads


class TestHeadPatternSimilarity:
    def test_self_similarity(self):
        """A head should have cosine similarity 1.0 with itself."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        sim = head_pattern_similarity(cache, 0, 0, 0, 0)
        np.testing.assert_allclose(sim, 1.0, atol=1e-5)

    def test_range(self):
        """Cosine similarity should be in [-1, 1]."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        sim = head_pattern_similarity(cache, 0, 0, 0, 1)
        assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5


class TestAttentionToToken:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn = attention_to_token(cache, layer=0, head=0, key_pos=0)
        assert attn.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn = attention_to_token(cache, layer=0, head=0, key_pos=0)
        assert np.all(attn >= 0)


class TestAttentionFromToken:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn = attention_from_token(cache, layer=0, head=0, query_pos=-1)
        assert attn.shape == (4,)

    def test_sums_to_one(self):
        """Attention from a position should sum to 1."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        attn = attention_from_token(cache, layer=0, head=0, query_pos=-1)
        np.testing.assert_allclose(np.sum(attn), 1.0, atol=1e-5)


class TestTopAttendedTokens:
    def test_returns_k_items(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        top = top_attended_tokens(cache, layer=0, head=0, query_pos=-1, k=3)
        assert len(top) == 3

    def test_sorted_descending(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        top = top_attended_tokens(cache, layer=0, head=0, query_pos=-1, k=4)
        weights = [w for _, w in top]
        assert weights == sorted(weights, reverse=True)


class TestCausalTracing:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        metric = lambda logits: float(logits[-1, 5])

        result = causal_tracing(model, tokens, corrupt_pos=1, metric_fn=metric)
        assert "clean" in result
        assert "corrupted" in result
        assert "restored_resid" in result
        assert "restored_attn" in result
        assert "restored_mlp" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        metric = lambda logits: float(logits[-1, 5])

        result = causal_tracing(model, tokens, corrupt_pos=1, metric_fn=metric)
        assert result["restored_resid"].shape == (model.cfg.n_layers,)
        assert result["restored_attn"].shape == (model.cfg.n_layers,)
        assert result["restored_mlp"].shape == (model.cfg.n_layers,)


class TestMaxAttentionPosition:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        positions = max_attention_position(cache, layer=0, head=0)
        assert positions.shape == (4,)

    def test_valid_positions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        positions = max_attention_position(cache, layer=0, head=0)
        assert np.all(positions >= 0)
        assert np.all(positions < 4)


class TestAttentionHeadSummary:
    def test_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        summary = attention_head_summary(cache, model)
        assert "entropy" in summary
        assert "max_attn" in summary
        assert "diag_score" in summary

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        cache = _get_cache(model, tokens)
        summary = attention_head_summary(cache, model)
        assert summary["entropy"].shape == (2, 4)
        assert summary["max_attn"].shape == (2, 4)
        assert summary["diag_score"].shape == (2, 4)
