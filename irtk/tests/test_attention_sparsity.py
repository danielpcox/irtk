"""Tests for attention_sparsity module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_sparsity import (
    attention_entropy_profile,
    attention_mass_distribution,
    sparse_vs_dense_heads,
    attention_window_analysis,
    attention_pattern_stability,
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


class TestAttentionEntropyProfile:
    def test_basic(self, model, tokens):
        result = attention_entropy_profile(model, tokens)
        assert "entropy" in result
        assert "mean_entropy" in result
        assert "sparsest_head" in result
        assert "densest_head" in result

    def test_shapes(self, model, tokens):
        result = attention_entropy_profile(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["entropy"].shape == (nl, nh, len(tokens))
        assert result["mean_entropy"].shape == (nl, nh)

    def test_entropy_nonneg(self, model, tokens):
        result = attention_entropy_profile(model, tokens)
        assert np.all(result["entropy"] >= 0)


class TestAttentionMassDistribution:
    def test_basic(self, model, tokens):
        result = attention_mass_distribution(model, tokens)
        assert "top_k_mass" in result
        assert "gini_coefficient" in result
        assert "max_attention" in result
        assert "effective_window" in result

    def test_shapes(self, model, tokens):
        result = attention_mass_distribution(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["top_k_mass"].shape == (nl, nh)
        assert result["gini_coefficient"].shape == (nl, nh)

    def test_mass_range(self, model, tokens):
        result = attention_mass_distribution(model, tokens)
        assert np.all(result["top_k_mass"] >= 0)
        assert np.all(result["top_k_mass"] <= 1.01)
        assert np.all(result["max_attention"] >= 0)
        assert np.all(result["max_attention"] <= 1.01)


class TestSparseVsDenseHeads:
    def test_basic(self, model, tokens):
        result = sparse_vs_dense_heads(model, tokens)
        assert "is_sparse" in result
        assert "sparsity_scores" in result
        assert "n_sparse" in result
        assert "n_dense" in result

    def test_counts_match(self, model, tokens):
        result = sparse_vs_dense_heads(model, tokens)
        total = model.cfg.n_layers * model.cfg.n_heads
        assert result["n_sparse"] + result["n_dense"] == total
        assert len(result["sparse_heads"]) == result["n_sparse"]
        assert len(result["dense_heads"]) == result["n_dense"]

    def test_sparsity_nonneg(self, model, tokens):
        result = sparse_vs_dense_heads(model, tokens)
        assert np.all(result["sparsity_scores"] >= 0)


class TestAttentionWindowAnalysis:
    def test_basic(self, model, tokens):
        result = attention_window_analysis(model, tokens)
        assert "mean_distance" in result
        assert "median_distance" in result
        assert "window_90" in result
        assert "local_fraction" in result

    def test_shapes(self, model, tokens):
        result = attention_window_analysis(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["mean_distance"].shape == (nl, nh)
        assert result["local_fraction"].shape == (nl, nh)

    def test_distance_nonneg(self, model, tokens):
        result = attention_window_analysis(model, tokens)
        assert np.all(result["mean_distance"] >= 0)
        assert np.all(result["local_fraction"] >= 0)


class TestAttentionPatternStability:
    def test_basic(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5]), jnp.array([10, 20, 30, 40, 49])]
        result = attention_pattern_stability(model, tokens_list)
        assert "pattern_variance" in result
        assert "stable_heads" in result
        assert "unstable_heads" in result
        assert "mean_pattern_norm" in result

    def test_shapes(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5])]
        result = attention_pattern_stability(model, tokens_list)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["pattern_variance"].shape == (nl, nh)

    def test_all_heads_classified(self, model, tokens):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5])]
        result = attention_pattern_stability(model, tokens_list)
        total = model.cfg.n_layers * model.cfg.n_heads
        assert len(result["stable_heads"]) + len(result["unstable_heads"]) == total
