"""Tests for token interaction analysis tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.token_interactions import (
    token_interaction_matrix,
    pairwise_synergy,
    conditional_attribution,
    bigram_attention_scores,
    token_pair_effect,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _metric(logits):
    return float(logits[-1, 0])


# ─── Token Interaction Matrix ────────────────────────────────────────────────


class TestTokenInteractionMatrix:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_interaction_matrix(model, tokens, _metric)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_interaction_matrix(model, tokens, _metric)
        assert np.all(result >= 0)

    def test_some_nonzero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_interaction_matrix(model, tokens, _metric)
        assert np.any(result > 0)

    def test_different_tokens_different_results(self):
        model = _make_model()
        r1 = token_interaction_matrix(model, jnp.array([0, 1, 2, 3]), _metric)
        r2 = token_interaction_matrix(model, jnp.array([10, 20, 30, 40]), _metric)
        assert not np.allclose(r1, r2, atol=1e-6)


# ─── Pairwise Synergy ────────────────────────────────────────────────────────


class TestPairwiseSynergy:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric)
        assert isinstance(result, dict)
        assert "synergy_matrix" in result
        assert "individual_effects" in result

    def test_synergy_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric)
        assert result["synergy_matrix"].shape == (4, 4)

    def test_synergy_symmetric(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric)
        S = result["synergy_matrix"]
        assert np.allclose(S, S.T, atol=1e-6)

    def test_synergy_diagonal_zero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric)
        assert np.allclose(np.diag(result["synergy_matrix"]), 0.0)

    def test_individual_effects_match(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric)
        # Individual effects should match token_interaction_matrix
        # (same metric, so absolute values should be close)
        interaction = token_interaction_matrix(model, tokens, _metric)
        # token_interaction_matrix uses abs, pairwise_synergy stores signed
        np.testing.assert_allclose(
            np.abs(result["individual_effects"]), interaction, atol=1e-5
        )

    def test_subset_positions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = pairwise_synergy(model, tokens, _metric, positions=[0, 2])
        assert result["synergy_matrix"].shape == (2, 2)
        assert result["individual_effects"].shape == (2,)


# ─── Conditional Attribution ─────────────────────────────────────────────────


class TestConditionalAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = conditional_attribution(model, tokens, _metric, target_pos=1, context_pos=2)
        assert isinstance(result, dict)
        assert "effect_target" in result
        assert "conditional_effect" in result
        assert "importance_change" in result

    def test_effect_both_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = conditional_attribution(model, tokens, _metric, target_pos=1, context_pos=2)
        # effect_both should be the combined effect
        # conditional_effect = effect_both - effect_context
        assert abs(
            result["conditional_effect"]
            - (result["effect_both"] - result["effect_context"])
        ) < 1e-5

    def test_importance_change_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = conditional_attribution(model, tokens, _metric, target_pos=1, context_pos=2)
        assert abs(
            result["importance_change"]
            - (result["conditional_effect"] - result["effect_target"])
        ) < 1e-5

    def test_same_position_self_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        # Same pos as target and context - effect_both should equal effect of ablating that pos
        result = conditional_attribution(model, tokens, _metric, target_pos=1, context_pos=1)
        assert abs(result["effect_both"] - result["effect_target"]) < 1e-5


# ─── Bigram Attention Scores ─────────────────────────────────────────────────


class TestBigramAttentionScores:
    def test_returns_matrix(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = bigram_attention_scores(model, tokens)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = bigram_attention_scores(model, tokens)
        assert np.all(result >= -1e-6)

    def test_single_layer(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        r_all = bigram_attention_scores(model, tokens)
        r0 = bigram_attention_scores(model, tokens, layer=0)
        r1 = bigram_attention_scores(model, tokens, layer=1)
        # Single layer result should differ from all-layers result
        assert r0.shape == (4, 4)
        assert r1.shape == (4, 4)

    def test_rows_approximately_sum_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        # For a single layer, the average pattern rows should sum to ~1
        result = bigram_attention_scores(model, tokens, layer=0)
        row_sums = result.sum(axis=-1)
        # Each row is average of softmax rows, so should sum to ~1
        np.testing.assert_allclose(row_sums, 1.0, atol=0.1)


# ─── Token Pair Effect ───────────────────────────────────────────────────────


class TestTokenPairEffect:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_pair_effect(model, tokens, _metric, pos_a=0, pos_b=1)
        assert isinstance(result, dict)
        assert "synergy" in result
        assert "redundancy_ratio" in result

    def test_synergy_consistent(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_pair_effect(model, tokens, _metric, pos_a=0, pos_b=1)
        expected_synergy = result["effect_both"] - result["effect_a"] - result["effect_b"]
        assert abs(result["synergy"] - expected_synergy) < 1e-5

    def test_redundancy_ratio_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_pair_effect(model, tokens, _metric, pos_a=0, pos_b=1)
        assert result["redundancy_ratio"] >= 0

    def test_clean_metric_matches(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = token_pair_effect(model, tokens, _metric, pos_a=0, pos_b=1)
        expected = _metric(model(tokens))
        assert abs(result["clean_metric"] - expected) < 1e-5
