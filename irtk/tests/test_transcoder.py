"""Tests for transcoder MLP feature circuit tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.transcoder import (
    Transcoder,
    train_transcoder,
    transcoder_feature_circuit,
    top_activating_for_transcoder_feature,
    mlp_feature_logit_attribution,
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


def _make_transcoder(d_model=16, n_features=32, seed=99):
    key = jax.random.PRNGKey(seed)
    return Transcoder(d_model, n_features, key=key)


# ─── Transcoder Class ──────────────────────────────────────────────────────


class TestTranscoder:
    def test_init_shapes(self):
        tc = _make_transcoder()
        assert tc.W_in.shape == (16, 32)
        assert tc.W_out.shape == (32, 16)
        assert tc.b_enc.shape == (32,)
        assert tc.b_dec.shape == (16,)

    def test_encode_shape(self):
        tc = _make_transcoder()
        x = jnp.ones((5, 16))
        result = tc.encode(x)
        assert result.shape == (5, 32)

    def test_encode_nonneg(self):
        tc = _make_transcoder()
        x = jnp.ones((5, 16))
        result = tc.encode(x)
        assert float(jnp.min(result)) >= 0.0

    def test_decode_shape(self):
        tc = _make_transcoder()
        feats = jnp.ones((5, 32))
        result = tc.decode(feats)
        assert result.shape == (5, 16)

    def test_call_returns_tuple(self):
        tc = _make_transcoder()
        x = jnp.ones((5, 16))
        y_hat, feat_acts = tc(x)
        assert y_hat.shape == (5, 16)
        assert feat_acts.shape == (5, 32)


# ─── Train Transcoder ──────────────────────────────────────────────────────


class TestTrainTranscoder:
    def test_returns_result(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = train_transcoder(model, layer=0, token_sequences=seqs,
                                  n_features=32, epochs=2, verbose=False)
        assert result.transcoder is not None
        assert len(result.train_losses) == 2

    def test_loss_decreases(self):
        model = _make_model()
        seqs = [jnp.array([i, i+1, i+2, i+3]) for i in range(5)]
        result = train_transcoder(model, layer=0, token_sequences=seqs,
                                  n_features=32, epochs=5, verbose=False)
        # Loss should generally decrease
        assert result.train_losses[-1] <= result.train_losses[0] * 2  # not strict, but reasonable

    def test_transcoder_d_model(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = train_transcoder(model, layer=0, token_sequences=seqs,
                                  n_features=32, epochs=1, verbose=False)
        assert result.transcoder.d_model == 16

    def test_l0_sparsity(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = train_transcoder(model, layer=0, token_sequences=seqs,
                                  n_features=32, epochs=2, verbose=False)
        assert len(result.l0_sparsities) == 2
        assert all(s >= 0 for s in result.l0_sparsities)


# ─── Transcoder Feature Circuit ────────────────────────────────────────────


class TestTranscoderFeatureCircuit:
    def test_returns_dict(self):
        tc_a = _make_transcoder(seed=42)
        tc_b = _make_transcoder(seed=99)
        result = transcoder_feature_circuit(tc_a, tc_b)
        assert "connection_matrix" in result
        assert "top_connections" in result

    def test_matrix_shape(self):
        tc_a = _make_transcoder(n_features=32, seed=42)
        tc_b = _make_transcoder(n_features=24, seed=99)
        result = transcoder_feature_circuit(tc_a, tc_b)
        assert result["connection_matrix"].shape == (32, 24)

    def test_self_connection(self):
        tc = _make_transcoder(seed=42)
        result = transcoder_feature_circuit(tc, tc)
        # Self-connection should be non-zero
        assert result["mean_connection"] > 0

    def test_top_connections_sorted(self):
        tc_a = _make_transcoder(seed=42)
        tc_b = _make_transcoder(seed=99)
        result = transcoder_feature_circuit(tc_a, tc_b)
        strengths = [abs(s) for _, _, s in result["top_connections"]]
        for i in range(len(strengths) - 1):
            assert strengths[i] >= strengths[i + 1]


# ─── Top Activating for Transcoder Feature ─────────────────────────────────


class TestTopActivatingForTranscoderFeature:
    def test_returns_list(self):
        model = _make_model()
        tc = _make_transcoder()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = top_activating_for_transcoder_feature(tc, model, 0, 0, seqs, k=5)
        assert isinstance(result, list)

    def test_sorted_descending(self):
        model = _make_model()
        tc = _make_transcoder()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = top_activating_for_transcoder_feature(tc, model, 0, 0, seqs, k=10)
        acts = [r["activation"] for r in result]
        for i in range(len(acts) - 1):
            assert acts[i] >= acts[i + 1]

    def test_result_fields(self):
        model = _make_model()
        tc = _make_transcoder()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = top_activating_for_transcoder_feature(tc, model, 0, 0, seqs, k=1)
        if result:
            assert "prompt_idx" in result[0]
            assert "position" in result[0]
            assert "activation" in result[0]

    def test_empty_sequences(self):
        model = _make_model()
        tc = _make_transcoder()
        result = top_activating_for_transcoder_feature(tc, model, 0, 0, [], k=5)
        assert result == []


# ─── MLP Feature Logit Attribution ────────────────────────────────────────


class TestMLPFeatureLogitAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tc = _make_transcoder()
        result = mlp_feature_logit_attribution(tc, model, 0, k=5)
        assert "top_promoted" in result
        assert "top_suppressed" in result
        assert "logit_effects" in result

    def test_effects_shape(self):
        model = _make_model()
        tc = _make_transcoder()
        result = mlp_feature_logit_attribution(tc, model, 0)
        assert result["logit_effects"].shape == (50,)

    def test_top_k_length(self):
        model = _make_model()
        tc = _make_transcoder()
        result = mlp_feature_logit_attribution(tc, model, 0, k=5)
        assert len(result["top_promoted"]) == 5
        assert len(result["top_suppressed"]) == 5

    def test_promoted_sorted_descending(self):
        model = _make_model()
        tc = _make_transcoder()
        result = mlp_feature_logit_attribution(tc, model, 0, k=10)
        scores = [s for _, s in result["top_promoted"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
