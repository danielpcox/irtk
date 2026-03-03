"""Tests for token_level_ablation module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_level_ablation import (
    per_token_knockout,
    token_necessity_sufficiency,
    minimal_token_set,
    pairwise_token_interaction,
    token_importance_ranking,
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
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


class TestPerTokenKnockout:
    def test_basic(self, model, tokens, metric_fn):
        result = per_token_knockout(model, tokens, metric_fn)
        assert "token_effects" in result
        assert "most_important_positions" in result
        assert "base_metric" in result

    def test_shape(self, model, tokens, metric_fn):
        result = per_token_knockout(model, tokens, metric_fn)
        assert result["token_effects"].shape == (len(tokens),)

    def test_ranking_length(self, model, tokens, metric_fn):
        result = per_token_knockout(model, tokens, metric_fn)
        assert len(result["most_important_positions"]) <= len(tokens)


class TestTokenNecessitySufficiency:
    def test_basic(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        assert "necessity" in result
        assert "sufficiency" in result
        assert "necessary_positions" in result
        assert "sufficient_positions" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        seq_len = len(tokens)
        assert result["necessity"].shape == (seq_len,)
        assert result["sufficiency"].shape == (seq_len,)

    def test_correlation_range(self, model, tokens, metric_fn):
        result = token_necessity_sufficiency(model, tokens, metric_fn)
        assert -1.01 <= result["necessity_sufficiency_correlation"] <= 1.01


class TestMinimalTokenSet:
    def test_basic(self, model, tokens, metric_fn):
        result = minimal_token_set(model, tokens, metric_fn)
        assert "minimal_set" in result
        assert "set_size" in result
        assert "metric_achieved" in result
        assert "coverage" in result

    def test_set_size(self, model, tokens, metric_fn):
        result = minimal_token_set(model, tokens, metric_fn)
        assert result["set_size"] <= len(tokens)
        assert result["set_size"] == len(result["minimal_set"])

    def test_trajectory(self, model, tokens, metric_fn):
        result = minimal_token_set(model, tokens, metric_fn)
        assert len(result["metric_trajectory"]) == result["set_size"]


class TestPairwiseTokenInteraction:
    def test_basic(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        assert "interaction_matrix" in result
        assert "synergistic_pairs" in result
        assert "redundant_pairs" in result
        assert "max_interaction" in result

    def test_shape(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        seq_len = len(tokens)
        assert result["interaction_matrix"].shape == (seq_len, seq_len)

    def test_symmetry(self, model, tokens, metric_fn):
        result = pairwise_token_interaction(model, tokens, metric_fn)
        mat = result["interaction_matrix"]
        assert np.allclose(mat, mat.T, atol=1e-6)


class TestTokenImportanceRanking:
    def test_basic(self, model, tokens, metric_fn):
        result = token_importance_ranking(model, tokens, metric_fn)
        assert "ranking" in result
        assert "importance_scores" in result
        assert "normalized_scores" in result
        assert "entropy" in result

    def test_ranking_complete(self, model, tokens, metric_fn):
        result = token_importance_ranking(model, tokens, metric_fn)
        assert len(result["ranking"]) == len(tokens)

    def test_normalized_sums(self, model, tokens, metric_fn):
        result = token_importance_ranking(model, tokens, metric_fn)
        assert abs(np.sum(result["normalized_scores"]) - 1.0) < 1e-5

    def test_leave_one_in(self, model, tokens, metric_fn):
        result = token_importance_ranking(model, tokens, metric_fn, method="leave_one_in")
        assert len(result["ranking"]) == len(tokens)

    def test_entropy_nonneg(self, model, tokens, metric_fn):
        result = token_importance_ranking(model, tokens, metric_fn)
        assert result["entropy"] >= 0
