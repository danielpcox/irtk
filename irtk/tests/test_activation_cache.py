"""Tests for ActivationCache."""

import jax.numpy as jnp
import pytest

from irtk.activation_cache import ActivationCache
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestActivationCache:
    def test_shorthand_access_attn(self):
        cache_dict = {"blocks.0.attn.hook_q": jnp.ones((5, 4, 4))}
        cache = ActivationCache(cache_dict)
        result = cache["q", 0]
        assert jnp.array_equal(result, jnp.ones((5, 4, 4)))

    def test_shorthand_access_mlp(self):
        cache_dict = {"blocks.1.mlp.hook_pre": jnp.ones((5, 64))}
        cache = ActivationCache(cache_dict)
        result = cache["pre", 1]
        assert result.shape == (5, 64)

    def test_shorthand_access_block(self):
        cache_dict = {"blocks.0.hook_resid_pre": jnp.ones((5, 16))}
        cache = ActivationCache(cache_dict)
        result = cache["resid_pre", 0]
        assert result.shape == (5, 16)

    def test_full_name_access(self):
        cache_dict = {"blocks.0.attn.hook_q": jnp.ones((5, 4, 4))}
        cache = ActivationCache(cache_dict)
        result = cache["blocks.0.attn.hook_q"]
        assert result.shape == (5, 4, 4)

    def test_missing_key_raises(self):
        cache = ActivationCache({})
        with pytest.raises(KeyError):
            _ = cache["nonexistent", 0]

    def test_contains(self):
        cache_dict = {"blocks.0.attn.hook_q": jnp.ones(3)}
        cache = ActivationCache(cache_dict)
        assert ("q", 0) in cache
        assert ("q", 1) not in cache
        assert "blocks.0.attn.hook_q" in cache

    def test_len_keys_values_items(self):
        cache_dict = {"a": jnp.ones(1), "b": jnp.ones(2)}
        cache = ActivationCache(cache_dict)
        assert len(cache) == 2
        assert set(cache.keys()) == {"a", "b"}

    def test_accumulated_resid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        resid = cache.accumulated_resid()
        # Should have: embed + n_layers resid_post = 3 components
        assert resid.shape[0] == 3
        assert resid.shape[1:] == (4, 16)

    def test_accumulated_resid_with_labels(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        _, cache = model.run_with_cache(tokens)
        resid, labels = cache.accumulated_resid(return_labels=True)
        assert len(labels) == resid.shape[0]

    def test_decompose_resid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        _, cache = model.run_with_cache(tokens)
        decomp = cache.decompose_resid()
        # embed + 2*(attn_out + mlp_out) = 5 components
        assert decomp.shape[0] == 5
        assert decomp.shape[1:] == (3, 16)

    def test_stack_activation(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        _, cache = model.run_with_cache(tokens)
        q_stack = cache.stack_activation("q")
        assert q_stack.shape[0] == 2  # n_layers
        assert q_stack.shape[1] == 3  # seq_len

    def test_logit_attrs(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        decomp = cache.decompose_resid()
        attrs = cache.logit_attrs(decomp)
        assert attrs.shape == (decomp.shape[0], 4, 50)

    def test_logit_attrs_with_tokens(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        decomp = cache.decompose_resid()
        attrs = cache.logit_attrs(decomp, tokens=tokens)
        assert attrs.shape == (decomp.shape[0], 4)
