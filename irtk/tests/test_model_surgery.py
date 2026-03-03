"""Tests for model surgery tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.model_surgery import (
    transplant_heads,
    transplant_mlp,
    knockout_head_weights,
    compare_heads_across_models,
    zero_out_layer,
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


class TestTransplantHeads:
    def test_returns_model(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_heads(donor, recipient, [(0, 0)])
        assert isinstance(result, HookedTransformer)

    def test_transplanted_head_matches_donor(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_heads(donor, recipient, [(0, 0)])
        # W_Q for head 0 should match donor
        assert np.allclose(result.blocks[0].attn.W_Q[0], donor.blocks[0].attn.W_Q[0])
        assert np.allclose(result.blocks[0].attn.W_O[0], donor.blocks[0].attn.W_O[0])

    def test_non_transplanted_head_unchanged(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_heads(donor, recipient, [(0, 0)])
        # Head 1 should be unchanged from recipient
        assert np.allclose(result.blocks[0].attn.W_Q[1], recipient.blocks[0].attn.W_Q[1])

    def test_transplant_multiple_heads(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_heads(donor, recipient, [(0, 0), (1, 2)])
        assert np.allclose(result.blocks[0].attn.W_Q[0], donor.blocks[0].attn.W_Q[0])
        assert np.allclose(result.blocks[1].attn.W_Q[2], donor.blocks[1].attn.W_Q[2])

    def test_transplant_changes_output(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = recipient(tokens)
        result = transplant_heads(donor, recipient, [(0, 0), (0, 1)])
        logits_after = result(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_original_unchanged(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        W_Q_orig = np.array(recipient.blocks[0].attn.W_Q[0])
        _ = transplant_heads(donor, recipient, [(0, 0)])
        # Original should be unchanged (immutable)
        assert np.allclose(recipient.blocks[0].attn.W_Q[0], W_Q_orig)


class TestTransplantMLP:
    def test_returns_model(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_mlp(donor, recipient, layer=0)
        assert isinstance(result, HookedTransformer)

    def test_transplanted_mlp_matches_donor(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_mlp(donor, recipient, layer=0)
        assert np.allclose(result.blocks[0].mlp.W_in, donor.blocks[0].mlp.W_in)
        assert np.allclose(result.blocks[0].mlp.W_out, donor.blocks[0].mlp.W_out)

    def test_other_layer_unchanged(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        result = transplant_mlp(donor, recipient, layer=0)
        assert np.allclose(result.blocks[1].mlp.W_in, recipient.blocks[1].mlp.W_in)

    def test_changes_output(self):
        donor = _make_model(seed=0)
        recipient = _make_model(seed=1)
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = recipient(tokens)
        result = transplant_mlp(donor, recipient, layer=0)
        logits_after = result(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)


class TestKnockoutHeadWeights:
    def test_returns_model(self):
        model = _make_model()
        result = knockout_head_weights(model, layer=0, head=0)
        assert isinstance(result, HookedTransformer)

    def test_W_O_zeroed(self):
        model = _make_model()
        result = knockout_head_weights(model, layer=0, head=0)
        assert np.allclose(result.blocks[0].attn.W_O[0], 0.0)

    def test_other_heads_unchanged(self):
        model = _make_model()
        result = knockout_head_weights(model, layer=0, head=0)
        assert np.allclose(result.blocks[0].attn.W_O[1], model.blocks[0].attn.W_O[1])

    def test_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        result = knockout_head_weights(model, layer=0, head=0)
        logits_after = result(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_knockout_all_heads_zeros_W_O(self):
        model = _make_model()
        # Knockout all heads in layer 0
        result = model
        for h in range(4):
            result = knockout_head_weights(result, layer=0, head=h)
        # All W_O should be zero
        assert np.allclose(result.blocks[0].attn.W_O, 0.0)


class TestCompareHeadsAcrossModels:
    def test_returns_dict(self):
        a = _make_model(seed=0)
        b = _make_model(seed=1)
        result = compare_heads_across_models(a, b, layer=0, head=0)
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        a = _make_model(seed=0)
        b = _make_model(seed=1)
        result = compare_heads_across_models(a, b, layer=0, head=0)
        for name in ["W_Q", "W_K", "W_V", "W_O"]:
            assert f"{name}_cosine" in result
            assert f"{name}_l2" in result
        assert "overall_cosine" in result

    def test_same_model_perfect_similarity(self):
        model = _make_model()
        result = compare_heads_across_models(model, model, layer=0, head=0)
        assert abs(result["W_Q_cosine"] - 1.0) < 1e-5
        assert abs(result["W_Q_l2"]) < 1e-5
        assert abs(result["overall_cosine"] - 1.0) < 1e-5

    def test_different_models_low_similarity(self):
        a = _make_model(seed=0)
        b = _make_model(seed=1)
        result = compare_heads_across_models(a, b, layer=0, head=0)
        # Random models should have near-zero cosine similarity
        assert abs(result["overall_cosine"]) < 0.5
        assert result["W_Q_l2"] > 0.0

    def test_cosine_in_range(self):
        a = _make_model(seed=0)
        b = _make_model(seed=1)
        result = compare_heads_across_models(a, b, layer=0, head=0)
        for name in ["W_Q", "W_K", "W_V", "W_O"]:
            assert -1.0 <= result[f"{name}_cosine"] <= 1.0


class TestZeroOutLayer:
    def test_returns_model(self):
        model = _make_model()
        result = zero_out_layer(model, layer=0)
        assert isinstance(result, HookedTransformer)

    def test_attn_W_O_zeroed(self):
        model = _make_model()
        result = zero_out_layer(model, layer=0, component="attn")
        assert np.allclose(result.blocks[0].attn.W_O, 0.0)
        # MLP should be unchanged
        assert np.allclose(result.blocks[0].mlp.W_out, model.blocks[0].mlp.W_out)

    def test_mlp_W_out_zeroed(self):
        model = _make_model()
        result = zero_out_layer(model, layer=0, component="mlp")
        assert np.allclose(result.blocks[0].mlp.W_out, 0.0)
        # Attn should be unchanged
        assert np.allclose(result.blocks[0].attn.W_O, model.blocks[0].attn.W_O)

    def test_both_zeroed(self):
        model = _make_model()
        result = zero_out_layer(model, layer=0, component="both")
        assert np.allclose(result.blocks[0].attn.W_O, 0.0)
        assert np.allclose(result.blocks[0].mlp.W_out, 0.0)

    def test_changes_output(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        result = zero_out_layer(model, layer=0)
        logits_after = result(tokens)
        assert not np.allclose(logits_before, logits_after, atol=1e-5)

    def test_other_layer_unchanged(self):
        model = _make_model()
        result = zero_out_layer(model, layer=0)
        assert np.allclose(result.blocks[1].attn.W_O, model.blocks[1].attn.W_O)
        assert np.allclose(result.blocks[1].mlp.W_out, model.blocks[1].mlp.W_out)
