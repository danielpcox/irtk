"""Tests for weight processing utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.weight_processing import fold_layer_norm, center_writing_weights, center_unembed


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestFoldLayerNorm:
    def test_logits_preserved(self):
        """Folding LN should produce identical logits."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        logits_before = model(tokens)

        folded = fold_layer_norm(model)
        logits_after = folded(tokens)

        np.testing.assert_allclose(
            np.array(logits_before), np.array(logits_after), atol=1e-4
        )

    def test_ln_weights_reset(self):
        """After folding, all LN w should be 1 and b should be 0."""
        model = _make_model()
        folded = fold_layer_norm(model)

        for l in range(model.cfg.n_layers):
            ln1 = folded.blocks[l].ln1
            if ln1 is not None:
                np.testing.assert_allclose(np.array(ln1.w), 1.0, atol=1e-6)
                if hasattr(ln1, 'b'):
                    np.testing.assert_allclose(np.array(ln1.b), 0.0, atol=1e-6)

            ln2 = folded.blocks[l].ln2
            if ln2 is not None:
                np.testing.assert_allclose(np.array(ln2.w), 1.0, atol=1e-6)
                if hasattr(ln2, 'b'):
                    np.testing.assert_allclose(np.array(ln2.b), 0.0, atol=1e-6)

        if folded.ln_final is not None:
            np.testing.assert_allclose(np.array(folded.ln_final.w), 1.0, atol=1e-6)


class TestCenterWritingWeights:
    def test_w_o_centered(self):
        """W_O should have zero mean along d_model axis."""
        model = _make_model()
        centered = center_writing_weights(model)

        for l in range(model.cfg.n_layers):
            W_O = centered.blocks[l].attn.W_O
            mean = jnp.mean(W_O, axis=-1)
            np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-6)

    def test_w_out_centered(self):
        """W_out should have zero mean along d_model axis."""
        model = _make_model()
        centered = center_writing_weights(model)

        for l in range(model.cfg.n_layers):
            W_out = centered.blocks[l].mlp.W_out
            mean = jnp.mean(W_out, axis=-1)
            np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-6)


class TestCenterUnembed:
    def test_w_u_centered(self):
        """W_U should have zero mean along vocab axis."""
        model = _make_model()
        centered = center_unembed(model)

        W_U = centered.unembed.W_U
        mean = jnp.mean(W_U, axis=-1)
        np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-6)

    def test_b_u_centered(self):
        """b_U should have zero mean."""
        model = _make_model()
        centered = center_unembed(model)
        np.testing.assert_allclose(float(jnp.mean(centered.unembed.b_U)), 0.0, atol=1e-6)

    def test_predictions_unchanged(self):
        """Centering unembed shouldn't change argmax predictions."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_before = model(tokens)
        preds_before = jnp.argmax(logits_before, axis=-1)

        centered = center_unembed(model)
        logits_after = centered(tokens)
        preds_after = jnp.argmax(logits_after, axis=-1)

        np.testing.assert_array_equal(np.array(preds_before), np.array(preds_after))
