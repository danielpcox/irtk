"""Tests for prompt_engineering_tools module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prompt_engineering_tools import (
    token_importance_map,
    attention_steering_analysis,
    critical_context_positions,
    prompt_sensitivity_map,
    prompt_comparison,
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


class TestTokenImportanceMap:
    def test_basic(self, model, tokens):
        result = token_importance_map(model, tokens)
        assert "importance_scores" in result
        assert "ranked_positions" in result
        assert "most_important" in result

    def test_scores_shape(self, model, tokens):
        result = token_importance_map(model, tokens)
        assert result["importance_scores"].shape == (len(tokens),)


class TestAttentionSteeringAnalysis:
    def test_basic(self, model, tokens):
        result = attention_steering_analysis(model, tokens)
        assert "steering_heads" in result
        assert "attention_distribution" in result
        assert "concentration_score" in result

    def test_heads_populated(self, model, tokens):
        result = attention_steering_analysis(model, tokens)
        assert len(result["steering_heads"]) == model.cfg.n_layers * model.cfg.n_heads


class TestCriticalContextPositions:
    def test_basic(self, model, tokens):
        result = critical_context_positions(model, tokens)
        assert "critical_positions" in result
        assert "position_scores" in result

    def test_scores_shape(self, model, tokens):
        result = critical_context_positions(model, tokens)
        assert result["position_scores"].shape == (len(tokens),)


class TestPromptSensitivityMap:
    def test_basic(self, model, tokens):
        result = prompt_sensitivity_map(model, tokens)
        assert "sensitivity_scores" in result
        assert "most_sensitive_positions" in result
        assert "mean_sensitivity" in result

    def test_scores_shape(self, model, tokens):
        result = prompt_sensitivity_map(model, tokens)
        assert result["sensitivity_scores"].shape == (len(tokens),)


class TestPromptComparison:
    def test_basic(self, model, tokens):
        tokens_b = jnp.array([25, 30, 35, 40, 45])
        result = prompt_comparison(model, tokens, tokens_b)
        assert "logit_diff" in result
        assert "attention_diff_per_layer" in result
        assert "most_different_layer" in result

    def test_layers_counted(self, model, tokens):
        tokens_b = jnp.array([25, 30, 35, 40, 45])
        result = prompt_comparison(model, tokens, tokens_b)
        assert len(result["residual_diff_per_layer"]) == model.cfg.n_layers
