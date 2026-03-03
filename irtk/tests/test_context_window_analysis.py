"""Tests for context_window_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.context_window_analysis import (
    effective_context_length,
    attention_decay_profile,
    position_dependent_capability,
    context_boundary_effects,
    information_horizon,
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


class TestEffectiveContextLength:
    def test_basic(self, model, tokens):
        result = effective_context_length(model, tokens)
        assert "per_layer" in result
        assert "overall_effective_length" in result
        assert "utilization_ratio" in result

    def test_utilization_range(self, model, tokens):
        result = effective_context_length(model, tokens)
        assert 0 <= result["utilization_ratio"] <= 1.0


class TestAttentionDecayProfile:
    def test_basic(self, model, tokens):
        result = attention_decay_profile(model, tokens)
        assert "per_layer" in result
        assert "mean_half_life" in result

    def test_all_layers(self, model, tokens):
        result = attention_decay_profile(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers


class TestPositionDependentCapability:
    def test_basic(self, model, tokens):
        result = position_dependent_capability(model, tokens)
        assert "per_position" in result
        assert "capability_trend" in result
        assert "mean_entropy" in result

    def test_per_position_count(self, model, tokens):
        result = position_dependent_capability(model, tokens)
        assert len(result["per_position"]) == len(tokens)


class TestContextBoundaryEffects:
    def test_basic(self, model, tokens):
        result = context_boundary_effects(model, tokens)
        assert "attention_across_boundary" in result
        assert "residual_discontinuity" in result

    def test_boundary_pos(self, model, tokens):
        result = context_boundary_effects(model, tokens, boundary_pos=2)
        assert result["boundary_pos"] == 2


class TestInformationHorizon:
    def test_basic(self, model, tokens):
        result = information_horizon(model, tokens)
        assert "per_position" in result
        assert "mean_horizon" in result

    def test_per_position_count(self, model, tokens):
        result = information_horizon(model, tokens)
        assert len(result["per_position"]) == len(tokens)
