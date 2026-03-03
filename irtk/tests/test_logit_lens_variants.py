"""Tests for logit_lens_variants module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_lens_variants import (
    contrastive_logit_lens,
    causal_logit_lens,
    residual_contribution_lens,
    token_lens_trajectory,
    logit_lens_difference,
)


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    m = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(m)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20])


class TestContrastiveLogitLens:
    def test_basic(self, model, tokens):
        tokens_b = jnp.array([1, 2, 3, 4, 5])
        result = contrastive_logit_lens(model, tokens, tokens_b)
        assert "layer_logit_diffs" in result
        assert "top_divergent_tokens" in result
        assert "divergence_per_layer" in result
        assert "convergence_layer" in result

    def test_shapes(self, model, tokens):
        tokens_b = jnp.array([1, 2, 3, 4, 5])
        result = contrastive_logit_lens(model, tokens, tokens_b)
        nl = model.cfg.n_layers
        d_vocab = model.cfg.d_vocab
        assert result["layer_logit_diffs"].shape == (nl, d_vocab)
        assert result["divergence_per_layer"].shape == (nl,)

    def test_convergence_valid(self, model, tokens):
        tokens_b = jnp.array([1, 2, 3, 4, 5])
        result = contrastive_logit_lens(model, tokens, tokens_b)
        assert 0 <= result["convergence_layer"] < model.cfg.n_layers


class TestCausalLogitLens:
    def test_basic(self, model, tokens):
        fn = lambda x: x * 0.5
        result = causal_logit_lens(model, tokens, 0, fn)
        assert "clean_predictions" in result
        assert "intervened_predictions" in result
        assert "prediction_shifts" in result
        assert "first_affected_layer" in result

    def test_shapes(self, model, tokens):
        fn = lambda x: x * 0.5
        result = causal_logit_lens(model, tokens, 0, fn, top_k=3)
        nl = model.cfg.n_layers
        assert result["clean_predictions"].shape == (nl, 3)
        assert result["intervened_predictions"].shape == (nl, 3)
        assert result["prediction_shifts"].shape == (nl,)

    def test_first_affected_valid(self, model, tokens):
        fn = lambda x: x * 0.5
        result = causal_logit_lens(model, tokens, 0, fn)
        assert 0 <= result["first_affected_layer"] < model.cfg.n_layers


class TestResidualContributionLens:
    def test_basic(self, model, tokens):
        result = residual_contribution_lens(model, tokens)
        assert "attn_logit_contributions" in result
        assert "mlp_logit_contributions" in result
        assert "cumulative_logits" in result
        assert "top_attn_tokens" in result
        assert "top_mlp_tokens" in result

    def test_shapes(self, model, tokens):
        result = residual_contribution_lens(model, tokens)
        nl = model.cfg.n_layers
        d_vocab = model.cfg.d_vocab
        assert result["attn_logit_contributions"].shape == (nl, d_vocab)
        assert result["mlp_logit_contributions"].shape == (nl, d_vocab)
        assert result["cumulative_logits"].shape == (nl, d_vocab)


class TestTokenLensTrajectory:
    def test_basic(self, model, tokens):
        result = token_lens_trajectory(model, tokens)
        assert "token_probs" in result
        assert "token_ranks" in result
        assert "emergence_layers" in result
        assert "final_prediction" in result

    def test_with_targets(self, model, tokens):
        result = token_lens_trajectory(model, tokens, target_tokens=[0, 1, 2])
        assert len(result["token_probs"]) == 3
        nl = model.cfg.n_layers
        for t in [0, 1, 2]:
            assert result["token_probs"][t].shape == (nl,)
            assert result["token_ranks"][t].shape == (nl,)

    def test_prob_range(self, model, tokens):
        result = token_lens_trajectory(model, tokens, target_tokens=[0])
        assert np.all(result["token_probs"][0] >= 0)
        assert np.all(result["token_probs"][0] <= 1.01)


class TestLogitLensDifference:
    def test_basic(self, model, tokens):
        result = logit_lens_difference(model, tokens, pos_a=0, pos_b=-1)
        assert "position_a_predictions" in result
        assert "position_b_predictions" in result
        assert "logit_diff_per_layer" in result
        assert "shared_top_tokens" in result

    def test_shapes(self, model, tokens):
        result = logit_lens_difference(model, tokens, pos_a=0, pos_b=-1, top_k=3)
        nl = model.cfg.n_layers
        assert result["position_a_predictions"].shape == (nl, 3)
        assert result["position_b_predictions"].shape == (nl, 3)
        assert result["logit_diff_per_layer"].shape == (nl,)
        assert result["shared_top_tokens"].shape == (nl,)

    def test_diff_nonneg(self, model, tokens):
        result = logit_lens_difference(model, tokens, pos_a=0, pos_b=-1)
        assert np.all(result["logit_diff_per_layer"] >= 0)
