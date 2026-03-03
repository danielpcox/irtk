"""Tests for token_erasure module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_erasure import (
    token_erasure_effects,
    token_necessity_sufficiency,
    erasure_curve,
    pairwise_token_interaction,
    layerwise_token_importance,
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


@pytest.fixture
def metric_fn():
    def fn(logits):
        return float(logits[-1, 0] - logits[-1, 1])
    return fn


class TestTokenErasureEffects:
    def test_basic(self, model, tokens, metric_fn):
        result = token_erasure_effects(model, tokens, metric_fn)
        assert "erasure_effects" in result
        assert "most_important_token" in result
        assert "least_important_token" in result
        assert "importance_ranking" in result
        assert "mean_effect" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = token_erasure_effects(model, tokens, metric_fn)
        seq_len = len(tokens)
        assert result["erasure_effects"].shape == (seq_len,)
        assert result["importance_ranking"].shape == (seq_len,)
        assert 0 <= result["most_important_token"] < seq_len
        assert 0 <= result["least_important_token"] < seq_len

    def test_effects_nonnegative(self, model, tokens, metric_fn):
        result = token_erasure_effects(model, tokens, metric_fn)
        assert np.all(result["erasure_effects"] >= 0)

    def test_ranking_is_permutation(self, model, tokens, metric_fn):
        result = token_erasure_effects(model, tokens, metric_fn)
        assert set(result["importance_ranking"]) == set(range(len(tokens)))


class TestTokenNecessitySufficiency:
    def test_basic(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        assert "necessity_scores" in result
        assert "sufficiency_scores" in result
        assert "necessary_tokens" in result
        assert "sufficient_tokens" in result
        assert "both_necessary_and_sufficient" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        seq_len = len(tokens)
        assert result["necessity_scores"].shape == (seq_len,)
        assert result["sufficiency_scores"].shape == (seq_len,)

    def test_scores_nonnegative(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        assert np.all(result["necessity_scores"] >= 0)
        assert np.all(result["sufficiency_scores"] >= 0)

    def test_both_is_intersection(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        both = result["both_necessary_and_sufficient"]
        for pos in both:
            assert pos in result["necessary_tokens"]
            assert pos in result["sufficient_tokens"]


class TestErasureCurve:
    def test_basic(self, model, tokens, metric_fn):
        result = erasure_curve(model, tokens, metric_fn)
        assert "n_erased" in result
        assert "metrics" in result
        assert "area_under_curve" in result
        assert "tokens_for_50pct_drop" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = erasure_curve(model, tokens, metric_fn)
        seq_len = len(tokens)
        assert len(result["n_erased"]) == seq_len + 1
        assert len(result["metrics"]) == seq_len + 1

    def test_first_metric_is_baseline(self, model, tokens, metric_fn):
        result = erasure_curve(model, tokens, metric_fn)
        baseline = metric_fn(model(tokens))
        np.testing.assert_allclose(result["metrics"][0], baseline, atol=1e-5)


class TestPairwiseTokenInteraction:
    def test_basic(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn, pos_a=0, pos_b=1)
        assert "effect_a" in result
        assert "effect_b" in result
        assert "effect_both" in result
        assert "interaction" in result
        assert "interaction_type" in result

    def test_effects_nonnegative(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        assert result["effect_a"] >= 0
        assert result["effect_b"] >= 0
        assert result["effect_both"] >= 0

    def test_interaction_type_valid(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        assert result["interaction_type"] in ("synergistic", "redundant", "independent")

    def test_interaction_formula(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        expected = result["effect_both"] - result["effect_a"] - result["effect_b"]
        np.testing.assert_allclose(result["interaction"], expected, atol=1e-6)


class TestLayerwiseTokenImportance:
    def test_basic(self, model, tokens, metric_fn):
        result = layerwise_token_importance(model, tokens, metric_fn)
        assert "importance_matrix" in result
        assert "emergence_layer" in result
        assert "peak_layer" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = layerwise_token_importance(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        seq_len = len(tokens)
        assert result["importance_matrix"].shape == (n_layers, seq_len)
        assert result["emergence_layer"].shape == (seq_len,)
        assert result["peak_layer"].shape == (seq_len,)

    def test_importance_nonnegative(self, model, tokens, metric_fn):
        result = layerwise_token_importance(model, tokens, metric_fn)
        assert np.all(result["importance_matrix"] >= 0)
