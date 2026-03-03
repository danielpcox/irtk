"""Tests for generation utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.generation import (
    top_k_sampling,
    nucleus_sampling,
    generate,
    generate_with_cache,
    generate_comparison,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestTopKSampling:
    def test_keeps_k_values(self):
        logits = jnp.arange(10, dtype=jnp.float32)
        filtered = top_k_sampling(logits, k=3)
        finite_count = jnp.sum(jnp.isfinite(filtered))
        assert int(finite_count) == 3

    def test_preserves_top_values(self):
        logits = jnp.array([1.0, 5.0, 3.0, 9.0, 2.0])
        filtered = top_k_sampling(logits, k=2)
        assert jnp.isfinite(filtered[1])  # 5.0
        assert jnp.isfinite(filtered[3])  # 9.0
        assert not jnp.isfinite(filtered[0])  # 1.0 should be -inf

    def test_k_equals_vocab(self):
        logits = jnp.arange(5, dtype=jnp.float32)
        filtered = top_k_sampling(logits, k=5)
        np.testing.assert_allclose(filtered, logits)


class TestNucleusSampling:
    def test_returns_valid_logits(self):
        logits = jnp.array([10.0, 1.0, 0.1, 0.01, -5.0])
        filtered = nucleus_sampling(logits, p=0.9)
        assert filtered.shape == logits.shape
        # The top token should definitely be kept
        assert jnp.isfinite(filtered[0])

    def test_high_p_keeps_more(self):
        logits = jnp.arange(10, dtype=jnp.float32)
        filtered_90 = nucleus_sampling(logits, p=0.9)
        filtered_50 = nucleus_sampling(logits, p=0.5)
        kept_90 = int(jnp.sum(jnp.isfinite(filtered_90)))
        kept_50 = int(jnp.sum(jnp.isfinite(filtered_50)))
        assert kept_90 >= kept_50

    def test_p_1_keeps_all(self):
        logits = jnp.arange(5, dtype=jnp.float32)
        filtered = nucleus_sampling(logits, p=1.0)
        assert jnp.all(jnp.isfinite(filtered))


class TestGenerate:
    def test_output_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate(model, tokens, max_new_tokens=5, temperature=0.0)
        assert len(result) == 8  # 3 + 5

    def test_preserves_prompt(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate(model, tokens, max_new_tokens=3, temperature=0.0)
        np.testing.assert_array_equal(result[:3], tokens)

    def test_valid_token_ids(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate(model, tokens, max_new_tokens=5, temperature=0.0)
        assert np.all(np.array(result) >= 0)
        assert np.all(np.array(result) < model.cfg.d_vocab)

    def test_greedy_deterministic(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result1 = generate(model, tokens, max_new_tokens=5, temperature=0.0)
        result2 = generate(model, tokens, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(result1, result2)

    def test_with_top_k(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate(model, tokens, max_new_tokens=3, temperature=1.0, top_k=5)
        assert len(result) == 6

    def test_with_top_p(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate(model, tokens, max_new_tokens=3, temperature=1.0, top_p=0.9)
        assert len(result) == 6

    def test_stop_token(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        # Generate greedy, find what token 4 would be, use it as stop
        full = generate(model, tokens, max_new_tokens=10, temperature=0.0)
        stop_tok = int(full[3])  # First generated token
        result = generate(model, tokens, max_new_tokens=10, temperature=0.0, stop_token=stop_tok)
        # Should stop after generating the stop token (prompt + 1 token)
        assert len(result) == 4


class TestGenerateWithCache:
    def test_output_length_and_caches(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result, caches = generate_with_cache(model, tokens, max_new_tokens=3)
        assert len(result) == 6  # 3 + 3
        assert len(caches) == 3  # one cache per step

    def test_caches_have_activations(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        _, caches = generate_with_cache(model, tokens, max_new_tokens=2)
        for cache in caches:
            assert len(cache.cache_dict) > 0

    def test_filter_hook_names(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        hook_names = ["blocks.0.hook_resid_post"]
        _, caches = generate_with_cache(
            model, tokens, max_new_tokens=2, hook_names=hook_names
        )
        for cache in caches:
            assert set(cache.cache_dict.keys()) == set(hook_names)


class TestGenerateComparison:
    def test_returns_expected_keys(self):
        model_a = _make_model()
        model_b = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate_comparison(model_a, model_b, tokens, max_new_tokens=3)
        assert "tokens_a" in result
        assert "tokens_b" in result
        assert "diverge_pos" in result

    def test_same_model_no_diverge(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = generate_comparison(model, model, tokens, max_new_tokens=3, temperature=0.0)
        assert result["diverge_pos"] == -1
        np.testing.assert_array_equal(result["tokens_a"], result["tokens_b"])
