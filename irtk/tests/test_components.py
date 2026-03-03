"""Tests for transformer components."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from irtk.hook_points import HookState
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.components.embed import Embed, PosEmbed
from irtk.components.unembed import Unembed
from irtk.components.layer_norm import LayerNorm, LayerNormPre, RMSNorm, RMSNormPre, get_layer_norm
from irtk.components.attention import Attention
from irtk.components.mlps import MLP, GatedMLP
from irtk.components.transformer_block import TransformerBlock


def _make_cfg(**overrides):
    defaults = dict(
        n_layers=2,
        d_model=32,
        n_ctx=64,
        d_head=8,
        n_heads=4,
        d_vocab=100,
        d_mlp=128,
        act_fn="gelu_new",
    )
    defaults.update(overrides)
    return HookedTransformerConfig(**defaults)


class TestEmbed:
    def test_output_shape(self):
        embed = Embed(d_vocab=100, d_model=32)
        tokens = jnp.array([0, 5, 10])
        result = embed(tokens)
        assert result.shape == (3, 32)

    def test_lookup_correctness(self):
        embed = Embed(d_vocab=10, d_model=4)
        W = jnp.arange(40, dtype=jnp.float32).reshape(10, 4)
        embed = eqx.tree_at(lambda m: m.W_E, embed, W)
        tokens = jnp.array([2, 5])
        result = embed(tokens)
        assert jnp.array_equal(result[0], W[2])
        assert jnp.array_equal(result[1], W[5])

    def test_hook_caching(self):
        embed = Embed(d_vocab=10, d_model=4)
        tokens = jnp.array([0, 1])
        cache = {}
        hs = HookState(cache=cache)
        embed(tokens, hs)
        assert "hook_embed" in cache
        assert cache["hook_embed"].shape == (2, 4)


class TestPosEmbed:
    def test_output_shape(self):
        pos = PosEmbed(n_ctx=64, d_model=32)
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = pos(tokens)
        assert result.shape == (5, 32)

    def test_slices_correctly(self):
        pos = PosEmbed(n_ctx=10, d_model=4)
        W = jnp.arange(40, dtype=jnp.float32).reshape(10, 4)
        pos = eqx.tree_at(lambda m: m.W_pos, pos, W)
        tokens = jnp.array([0, 1, 2])
        result = pos(tokens)
        assert jnp.array_equal(result, W[:3])


class TestUnembed:
    def test_output_shape(self):
        unembed = Unembed(d_model=32, d_vocab_out=100)
        x = jnp.ones((5, 32))
        result = unembed(x)
        assert result.shape == (5, 100)

    def test_linear_operation(self):
        unembed = Unembed(d_model=4, d_vocab_out=3)
        W_U = jnp.eye(4)[:, :3]
        unembed = eqx.tree_at(lambda m: m.W_U, unembed, W_U)
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        result = unembed(x)
        assert jnp.allclose(result, jnp.array([[1.0, 2.0, 3.0]]))


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(d_model=32, name_prefix="test.")
        x = jnp.ones((5, 32))
        result = ln(x)
        assert result.shape == (5, 32)

    def test_normalization(self):
        ln = LayerNorm(d_model=4, name_prefix="test.")
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        result = ln(x)
        # After norm with w=1, b=0: should be centered and unit variance
        assert jnp.allclose(result.mean(axis=-1), 0.0, atol=1e-5)
        assert jnp.allclose(result.var(axis=-1), 1.0, atol=1e-1)

    def test_hook_caching(self):
        ln = LayerNorm(d_model=4, name_prefix="ln.")
        cache = {}
        hs = HookState(cache=cache)
        x = jnp.ones((3, 4))
        ln(x, hs)
        assert "ln.hook_scale" in cache
        assert "ln.hook_normalized" in cache


class TestLayerNormPre:
    def test_no_params(self):
        ln = LayerNormPre(eps=1e-5, name_prefix="test.")
        # Should not have w or b
        assert not hasattr(ln, "w")


class TestRMSNorm:
    def test_output_shape(self):
        rms = RMSNorm(d_model=32, name_prefix="test.")
        x = jnp.ones((5, 32))
        result = rms(x)
        assert result.shape == (5, 32)

    def test_rms_normalization(self):
        rms = RMSNorm(d_model=4, name_prefix="test.")
        x = jnp.array([[2.0, 2.0, 2.0, 2.0]])
        result = rms(x)
        # RMS of [2,2,2,2] = 2. After norm: [1,1,1,1] * w(=1) = [1,1,1,1]
        assert jnp.allclose(result, jnp.ones((1, 4)), atol=1e-5)


class TestGetLayerNorm:
    def test_returns_correct_type(self):
        assert isinstance(get_layer_norm("LN", 32, name_prefix=""), LayerNorm)
        assert isinstance(get_layer_norm("LNPre", 32, name_prefix=""), LayerNormPre)
        assert isinstance(get_layer_norm("RMS", 32, name_prefix=""), RMSNorm)
        assert isinstance(get_layer_norm("RMSPre", 32, name_prefix=""), RMSNormPre)

    def test_returns_none(self):
        assert get_layer_norm(None, 32, name_prefix="") is None

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError):
            get_layer_norm("unknown", 32, name_prefix="")


class TestAttention:
    def test_output_shape(self):
        cfg = _make_cfg()
        attn = Attention(cfg, layer_idx=0)
        x = jnp.ones((10, cfg.d_model))
        result = attn(x)
        assert result.shape == (10, cfg.d_model)

    def test_hook_caching(self):
        cfg = _make_cfg()
        attn = Attention(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        attn(x, hs)
        assert "blocks.0.attn.hook_q" in cache
        assert "blocks.0.attn.hook_k" in cache
        assert "blocks.0.attn.hook_v" in cache
        assert "blocks.0.attn.hook_z" in cache
        assert "blocks.0.attn.hook_attn_scores" in cache
        assert "blocks.0.attn.hook_pattern" in cache
        assert "blocks.0.attn.hook_result" in cache

    def test_cached_shapes(self):
        cfg = _make_cfg(n_heads=4, d_head=8, d_model=32)
        attn = Attention(cfg, layer_idx=0)
        x = jnp.ones((6, 32))
        cache = {}
        hs = HookState(cache=cache)
        attn(x, hs)
        assert cache["blocks.0.attn.hook_q"].shape == (6, 4, 8)
        assert cache["blocks.0.attn.hook_k"].shape == (6, 4, 8)
        assert cache["blocks.0.attn.hook_v"].shape == (6, 4, 8)
        assert cache["blocks.0.attn.hook_attn_scores"].shape == (4, 6, 6)
        assert cache["blocks.0.attn.hook_pattern"].shape == (4, 6, 6)
        assert cache["blocks.0.attn.hook_z"].shape == (6, 4, 8)
        assert cache["blocks.0.attn.hook_result"].shape == (6, 32)

    def test_causal_mask(self):
        """Attention pattern should be lower-triangular for causal attention."""
        cfg = _make_cfg(attention_dir="causal")
        attn = Attention(cfg, layer_idx=0)
        # Use random weights so attention isn't all zeros
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        attn = eqx.tree_at(lambda m: m.W_Q, attn, jax.random.normal(keys[0], attn.W_Q.shape) * 0.1)
        attn = eqx.tree_at(lambda m: m.W_K, attn, jax.random.normal(keys[1], attn.W_K.shape) * 0.1)
        attn = eqx.tree_at(lambda m: m.W_V, attn, jax.random.normal(keys[2], attn.W_V.shape) * 0.1)
        attn = eqx.tree_at(lambda m: m.W_O, attn, jax.random.normal(keys[3], attn.W_O.shape) * 0.1)

        x = jnp.ones((4, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        attn(x, hs)
        pattern = cache["blocks.0.attn.hook_pattern"]
        # Upper triangle should be zero (causal)
        for h in range(cfg.n_heads):
            for i in range(4):
                for j in range(i + 1, 4):
                    assert pattern[h, i, j] < 1e-6

    def test_gqa(self):
        cfg = _make_cfg(n_heads=4, n_key_value_heads=2, d_head=8, d_model=32)
        attn = Attention(cfg, layer_idx=0)
        x = jnp.ones((5, 32))
        result = attn(x)
        assert result.shape == (5, 32)

    def test_rotary(self):
        cfg = _make_cfg(positional_embedding_type="rotary", rotary_dim=8)
        attn = Attention(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = attn(x)
        assert result.shape == (5, cfg.d_model)


class TestMLP:
    def test_output_shape(self):
        cfg = _make_cfg()
        mlp = MLP(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = mlp(x)
        assert result.shape == (5, cfg.d_model)

    def test_hook_caching(self):
        cfg = _make_cfg()
        mlp = MLP(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        mlp(x, hs)
        assert "blocks.0.mlp.hook_pre" in cache
        assert "blocks.0.mlp.hook_post" in cache
        assert cache["blocks.0.mlp.hook_pre"].shape == (5, cfg.d_mlp)


class TestGatedMLP:
    def test_output_shape(self):
        cfg = _make_cfg(gated_mlp=True, act_fn="silu")
        mlp = GatedMLP(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = mlp(x)
        assert result.shape == (5, cfg.d_model)

    def test_hook_caching(self):
        cfg = _make_cfg(gated_mlp=True, act_fn="silu")
        mlp = GatedMLP(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        mlp(x, hs)
        assert "blocks.0.mlp.hook_pre" in cache
        assert "blocks.0.mlp.hook_post" in cache
        assert "blocks.0.mlp.hook_pre_linear" in cache


class TestTransformerBlock:
    def test_sequential_output_shape(self):
        cfg = _make_cfg()
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = block(x)
        assert result.shape == (5, cfg.d_model)

    def test_parallel_output_shape(self):
        cfg = _make_cfg(parallel_attn_mlp=True)
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = block(x)
        assert result.shape == (5, cfg.d_model)

    def test_residual_hooks(self):
        cfg = _make_cfg()
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        block(x, hs)
        assert "blocks.0.hook_resid_pre" in cache
        assert "blocks.0.hook_resid_mid" in cache
        assert "blocks.0.hook_resid_post" in cache
        assert "blocks.0.hook_attn_out" in cache
        assert "blocks.0.hook_mlp_out" in cache

    def test_parallel_no_resid_mid(self):
        """Parallel mode skips hook_resid_mid (there's no 'mid' point)."""
        cfg = _make_cfg(parallel_attn_mlp=True)
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        cache = {}
        hs = HookState(cache=cache)
        block(x, hs)
        # In parallel mode, hook_resid_mid fires but is just the identity
        # (it's called on a value that never gets used in the residual path)
        # Actually looking at the code, hook_resid_mid is NOT called in parallel mode
        assert "blocks.0.hook_resid_mid" not in cache

    def test_no_norm(self):
        cfg = _make_cfg(normalization_type=None)
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        result = block(x)
        assert result.shape == (5, cfg.d_model)

    def test_jit_without_hooks(self):
        cfg = _make_cfg()
        block = TransformerBlock(cfg, layer_idx=0)
        x = jnp.ones((5, cfg.d_model))
        jitted = eqx.filter_jit(block)
        result = jitted(x)
        assert result.shape == (5, cfg.d_model)
