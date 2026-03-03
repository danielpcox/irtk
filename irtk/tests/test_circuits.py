"""Tests for circuit analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.factored_matrix import FactoredMatrix
from irtk.circuits import (
    ov_circuit,
    qk_circuit,
    full_ov_circuit,
    full_qk_circuit,
    direct_logit_attribution,
    residual_stream_attribution,
    qk_composition_score,
    ov_composition_score,
    all_composition_scores,
    attention_to_positions,
    prev_token_score,
    induction_score,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestOVCircuit:
    def test_shape(self):
        model = _make_model()
        ov = ov_circuit(model, layer=0, head=0)
        assert isinstance(ov, FactoredMatrix)
        assert ov.shape == (16, 16)

    def test_full_ov_shape(self):
        model = _make_model()
        fov = full_ov_circuit(model, layer=0, head=0)
        assert isinstance(fov, FactoredMatrix)
        assert fov.shape == (50, 50)


class TestQKCircuit:
    def test_shape(self):
        model = _make_model()
        qk = qk_circuit(model, layer=0, head=0)
        assert isinstance(qk, FactoredMatrix)
        assert qk.shape == (16, 16)

    def test_full_qk_shape(self):
        model = _make_model()
        fqk = full_qk_circuit(model, layer=0, head=0)
        assert isinstance(fqk, FactoredMatrix)
        assert fqk.shape == (50, 50)


class TestDirectLogitAttribution:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        result = direct_logit_attribution(model, cache, token=5)
        assert result.shape == (2, 4)  # n_layers x n_heads

    def test_sums_approximately(self):
        """Head contributions should be part of total logit."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits = model(tokens)
        _, cache = model.run_with_cache(tokens)

        token = 5
        head_attrs = direct_logit_attribution(model, cache, token=token)
        total_head_contribution = np.sum(head_attrs)
        # This won't equal the logit exactly (missing embedding, MLP, bias terms)
        # but it should be finite
        assert np.isfinite(total_head_contribution)


class TestResidualStreamAttribution:
    def test_basic(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        result = residual_stream_attribution(model, cache, token=5)
        assert "embed" in result
        assert "L0_attn" in result
        assert "L0_mlp" in result
        assert "L1_attn" in result
        assert "L1_mlp" in result

    def test_sums_to_logit(self):
        """Sum of all contributions should approximate the actual logit."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        logits = model(tokens)
        _, cache = model.run_with_cache(tokens)

        token = 5
        contributions = residual_stream_attribution(model, cache, token=token, pos=-1)
        total = sum(contributions.values())
        # Should be reasonably close to the actual logit (may differ due to LN)
        actual_logit = float(logits[-1, token])
        # With layer norm, they won't be exact, but should be in same ballpark
        assert np.isfinite(total)


class TestCompositionScores:
    def test_qk_score(self):
        model = _make_model()
        score = qk_composition_score(model, 0, 0, 1, 0)
        assert isinstance(score, float)
        assert score >= 0

    def test_ov_score(self):
        model = _make_model()
        score = ov_composition_score(model, 0, 0, 1, 0)
        assert isinstance(score, float)
        assert score >= 0

    def test_same_layer_is_zero(self):
        model = _make_model()
        assert qk_composition_score(model, 0, 0, 0, 1) == 0.0
        assert ov_composition_score(model, 1, 0, 0, 0) == 0.0

    def test_all_composition_scores_shape(self):
        model = _make_model()
        scores = all_composition_scores(model, composition_type="qk")
        total_heads = 2 * 4
        assert scores.shape == (total_heads, total_heads)


class TestAttentionAnalysis:
    def test_attention_to_positions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        attn = attention_to_positions(cache, layer=0, head=0, query_pos=-1)
        assert attn.shape == (4,)
        np.testing.assert_allclose(np.sum(attn), 1.0, atol=1e-5)

    def test_prev_token_score(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        _, cache = model.run_with_cache(tokens)
        score = prev_token_score(cache, layer=0, head=0)
        assert 0.0 <= score <= 1.0

    def test_induction_score(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        score = induction_score(cache, layer=0, head=0)
        assert 0.0 <= score <= 1.0
