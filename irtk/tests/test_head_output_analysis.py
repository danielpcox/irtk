"""Tests for head_output_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_output_analysis import (
    head_output_decomposition,
    value_weighted_analysis,
    output_direction_characterization,
    head_cooperation_competition,
    output_norm_analysis,
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


class TestHeadOutputDecomposition:
    def test_basic(self, model, tokens):
        result = head_output_decomposition(model, tokens, layer=0)
        assert "per_head" in result
        assert result["n_heads"] == model.cfg.n_heads

    def test_per_head_keys(self, model, tokens):
        result = head_output_decomposition(model, tokens, layer=0)
        for h_info in result["per_head"]:
            assert "output_norm" in h_info
            assert "promoted" in h_info


class TestValueWeightedAnalysis:
    def test_basic(self, model, tokens):
        result = value_weighted_analysis(model, tokens, layer=0, head=0)
        assert "per_source_contribution" in result
        assert "attention_weights" in result
        assert "dominant_source" in result

    def test_weights_shape(self, model, tokens):
        result = value_weighted_analysis(model, tokens, layer=0, head=0)
        assert result["attention_weights"].shape == (len(tokens),)


class TestOutputDirectionCharacterization:
    def test_basic(self, model, tokens):
        result = output_direction_characterization(model, tokens, layer=0)
        assert "pairwise_cosines" in result
        assert "diversity_score" in result

    def test_diversity_range(self, model, tokens):
        result = output_direction_characterization(model, tokens, layer=0)
        assert -0.5 <= result["diversity_score"] <= 1.5


class TestHeadCooperationCompetition:
    def test_basic(self, model, tokens):
        result = head_cooperation_competition(model, tokens, layer=0)
        assert "cooperation_matrix" in result
        assert "cooperating_pairs" in result
        assert "competing_pairs" in result

    def test_matrix_shape(self, model, tokens):
        result = head_cooperation_competition(model, tokens, layer=0)
        nh = model.cfg.n_heads
        assert result["cooperation_matrix"].shape == (nh, nh)


class TestOutputNormAnalysis:
    def test_basic(self, model, tokens):
        result = output_norm_analysis(model, tokens, layer=0)
        assert "norm_matrix" in result
        assert "head_mean_norms" in result
        assert "max_norm_head" in result

    def test_shapes(self, model, tokens):
        result = output_norm_analysis(model, tokens, layer=0)
        assert result["norm_matrix"].shape == (model.cfg.n_heads, len(tokens))
        assert result["head_mean_norms"].shape == (model.cfg.n_heads,)
