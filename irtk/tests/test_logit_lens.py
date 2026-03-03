"""Tests for logit lens and tuned lens."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.logit_lens import (
    logit_lens,
    logit_lens_top_k,
    logit_lens_correct_prob,
    logit_lens_kl_divergence,
    TunedLensProbe,
    TunedLens,
    train_tuned_lens,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestLogitLens:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens(model, tokens)
        # n_layers+1 components (embed + 2 layers), seq_len=4, d_vocab=50
        assert result.shape == (3, 4, 50)

    def test_output_shape_no_ln(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens(model, tokens, apply_ln=False)
        assert result.shape == (3, 4, 50)

    def test_return_probs(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        probs = logit_lens(model, tokens, return_probs=True)
        # Probabilities should sum to ~1 along vocab axis
        sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_final_layer_matches_model(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits_model = model(tokens)
        logits_lens = logit_lens(model, tokens, apply_ln=True)
        # The last component should match the model's output
        np.testing.assert_allclose(
            logits_lens[-1], np.array(logits_model), atol=1e-5
        )


class TestLogitLensTopK:
    def test_structure(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        result = logit_lens_top_k(model, tokens, k=3)
        assert len(result) == 3  # n_components
        assert len(result[0]) == 3  # seq_len
        assert len(result[0][0]) == 3  # top-k
        # Each entry is (token_id, probability)
        tok_id, prob = result[0][0][0]
        assert isinstance(tok_id, int)
        assert 0.0 <= prob <= 1.0


class TestLogitLensCorrectProb:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens_correct_prob(model, tokens)
        # [n_components, seq_len - 1]
        assert result.shape == (3, 3)

    def test_probabilities_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens_correct_prob(model, tokens)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestLogitLensKLDivergence:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens_kl_divergence(model, tokens)
        # [n_components - 1, seq_len] (excludes final)
        assert result.shape == (2, 4)

    def test_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_lens_kl_divergence(model, tokens)
        # KL divergence is non-negative
        assert np.all(result >= -1e-6)  # small tolerance for numerics


class TestTunedLensProbe:
    def test_identity_init(self):
        key = jax.random.PRNGKey(0)
        probe = TunedLensProbe(16, key=key)
        x = jnp.ones((5, 16))
        out = probe(x)
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-6)

    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        probe = TunedLensProbe(16, key=key)
        x = jnp.ones((5, 16))
        assert probe(x).shape == (5, 16)


class TestTunedLens:
    def test_apply_shape(self):
        model = _make_model()
        key = jax.random.PRNGKey(0)
        probes = [TunedLensProbe(16, key=key) for _ in range(3)]
        tl = TunedLens(probes=probes, model=model)
        tokens = jnp.array([0, 1, 2, 3])
        result = tl.apply(tokens)
        assert result.shape == (3, 4, 50)

    def test_untrained_matches_logit_lens(self):
        """Untrained tuned lens (identity probes) should match logit lens."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        key = jax.random.PRNGKey(0)
        probes = [TunedLensProbe(16, key=key) for _ in range(3)]
        tl = TunedLens(probes=probes, model=model)

        ll_result = logit_lens(model, tokens, apply_ln=True)
        tl_result = tl.apply(tokens, apply_ln=True)
        np.testing.assert_allclose(tl_result, ll_result, atol=1e-5)


class TestTrainTunedLens:
    def test_basic(self):
        model = _make_model()
        tokens = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        result = train_tuned_lens(model, tokens, epochs=3, verbose=False)
        assert len(result.tuned_lens.probes) == 2  # n_components - 1
        assert len(result.train_losses) == 2
        assert len(result.train_losses[0]) == 3  # epochs

    def test_single_sequence(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = train_tuned_lens(model, tokens, epochs=2, verbose=False)
        assert result.tuned_lens is not None

    def test_with_validation(self):
        model = _make_model()
        train_tokens = jnp.array([[0, 1, 2, 3]])
        val_tokens = jnp.array([[4, 5, 6, 7]])
        result = train_tuned_lens(
            model, train_tokens, val_tokens=val_tokens,
            epochs=3, verbose=False,
        )
        assert len(result.val_losses[0]) == 3
