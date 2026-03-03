"""Tests for sparse feature interpretation tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.sparse_features import (
    feature_activation_examples,
    feature_to_feature_correlation,
    feature_circuit,
    feature_token_bias,
    feature_downstream_effect,
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


def _make_sae(d_model=16, n_features=32, seed=99):
    key = jax.random.PRNGKey(seed)
    return SparseAutoencoder(d_model, n_features, key=key)


# ─── Feature Activation Examples ────────────────────────────────────────────


class TestFeatureActivationExamples:
    def test_returns_list(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = feature_activation_examples(sae, model, seqs, 0, "blocks.0.hook_resid_post", k=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_sorted_descending(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = feature_activation_examples(sae, model, seqs, 0, "blocks.0.hook_resid_post", k=10)
        acts = [r["activation"] for r in result]
        for i in range(len(acts) - 1):
            assert acts[i] >= acts[i + 1]

    def test_result_fields(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_activation_examples(sae, model, seqs, 0, "blocks.0.hook_resid_post", k=1)
        if result:
            assert "prompt_idx" in result[0]
            assert "position" in result[0]
            assert "activation" in result[0]

    def test_empty_sequences(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_activation_examples(sae, model, [], 0, "blocks.0.hook_resid_post")
        assert result == []


# ─── Feature-to-Feature Correlation ─────────────────────────────────────────


class TestFeatureToFeatureCorrelation:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_to_feature_correlation(sae, model, seqs, 0, 1, "blocks.0.hook_resid_post")
        assert "correlation" in result
        assert "co_activation_rate" in result

    def test_self_correlation_is_one(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = feature_to_feature_correlation(sae, model, seqs, 0, 0, "blocks.0.hook_resid_post")
        assert abs(result["correlation"] - 1.0) < 0.01 or abs(result["correlation"]) < 0.01  # may be 0 if constant

    def test_rates_in_range(self):
        model = _make_model()
        sae = _make_sae()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_to_feature_correlation(sae, model, seqs, 0, 1, "blocks.0.hook_resid_post")
        assert 0 <= result["co_activation_rate"] <= 1.0

    def test_empty_sequences(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_to_feature_correlation(sae, model, [], 0, 1, "blocks.0.hook_resid_post")
        assert result["correlation"] == 0.0


# ─── Feature Circuit ────────────────────────────────────────────────────────


class TestFeatureCircuit:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_circuit(sae, model, tokens, 0, "blocks.1.hook_resid_post")
        assert "clean_activation" in result
        assert "head_effects" in result
        assert "mlp_effects" in result

    def test_clean_activation_type(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_circuit(sae, model, tokens, 0, "blocks.1.hook_resid_post")
        assert isinstance(result["clean_activation"], float)

    def test_circuit_components_sorted(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_circuit(sae, model, tokens, 0, "blocks.1.hook_resid_post", threshold=0.0)
        comps = result["circuit_components"]
        for i in range(len(comps) - 1):
            assert abs(comps[i][1]) >= abs(comps[i + 1][1])


# ─── Feature Token Bias ────────────────────────────────────────────────────


class TestFeatureTokenBias:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_token_bias(sae, model, 0, k=5)
        assert "top_positive" in result
        assert "top_negative" in result
        assert "all_scores" in result

    def test_score_shape(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_token_bias(sae, model, 0)
        assert result["all_scores"].shape == (50,)  # d_vocab=50

    def test_top_k_length(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_token_bias(sae, model, 0, k=5)
        assert len(result["top_positive"]) == 5
        assert len(result["top_negative"]) == 5

    def test_top_positive_descending(self):
        model = _make_model()
        sae = _make_sae()
        result = feature_token_bias(sae, model, 0, k=5)
        scores = [s for _, s in result["top_positive"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ─── Feature Downstream Effect ──────────────────────────────────────────────


class TestFeatureDownstreamEffect:
    def test_returns_dict(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_downstream_effect(sae, model, tokens, 0, "blocks.0.hook_resid_post", k=5)
        assert "top_promoted" in result
        assert "top_suppressed" in result
        assert "logit_diff_norm" in result

    def test_logit_diff_nonneg(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_downstream_effect(sae, model, tokens, 0, "blocks.0.hook_resid_post")
        assert result["logit_diff_norm"] >= 0

    def test_top_k_length(self):
        model = _make_model()
        sae = _make_sae()
        tokens = jnp.array([0, 1, 2, 3])
        result = feature_downstream_effect(sae, model, tokens, 0, "blocks.0.hook_resid_post", k=3)
        assert len(result["top_promoted"]) == 3
        assert len(result["top_suppressed"]) == 3
