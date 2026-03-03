"""Tests for logit_decomposition_advanced module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_decomposition_advanced import (
    per_component_logit_contribution,
    logit_interaction_terms,
    direct_vs_indirect_logit,
    logit_lens_decomposition,
    token_specific_attribution,
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


class TestPerComponentLogitContribution:
    def test_basic(self, model, tokens):
        result = per_component_logit_contribution(model, tokens)
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "embed_contribution" in result
        assert "total_logit" in result

    def test_shapes(self, model, tokens):
        result = per_component_logit_contribution(model, tokens)
        nl = model.cfg.n_layers
        assert result["attn_contributions"].shape == (nl,)
        assert result["mlp_contributions"].shape == (nl,)


class TestLogitInteractionTerms:
    def test_basic(self, model, tokens):
        result = logit_interaction_terms(model, tokens)
        assert "interaction_matrix" in result
        assert "single_effects" in result
        assert "components" in result

    def test_matrix_shape(self, model, tokens):
        result = logit_interaction_terms(model, tokens)
        n = model.cfg.n_layers * 2  # attn + mlp per layer
        assert result["interaction_matrix"].shape == (n, n)


class TestDirectVsIndirectLogit:
    def test_basic(self, model, tokens):
        result = direct_vs_indirect_logit(model, tokens)
        assert "direct_contributions" in result
        assert "indirect_contributions" in result
        assert "direct_fraction" in result
        assert "indirect_fraction" in result

    def test_fractions_sum(self, model, tokens):
        result = direct_vs_indirect_logit(model, tokens)
        total = result["direct_fraction"] + result["indirect_fraction"]
        assert abs(total - 1.0) < 0.01


class TestLogitLensDecomposition:
    def test_basic(self, model, tokens):
        result = logit_lens_decomposition(model, tokens)
        assert "logit_trajectory" in result
        assert "delta_per_layer" in result
        assert "attn_delta" in result
        assert "mlp_delta" in result

    def test_shapes(self, model, tokens):
        result = logit_lens_decomposition(model, tokens)
        nl = model.cfg.n_layers
        assert result["logit_trajectory"].shape == (nl,)
        assert result["attn_delta"].shape == (nl,)
        assert result["mlp_delta"].shape == (nl,)


class TestTokenSpecificAttribution:
    def test_basic(self, model, tokens):
        result = token_specific_attribution(model, tokens)
        assert "per_target" in result
        assert "target_tokens" in result
        assert len(result["target_tokens"]) > 0

    def test_position_effects(self, model, tokens):
        result = token_specific_attribution(model, tokens)
        for target, info in result["per_target"].items():
            assert info["position_effects"].shape == (len(tokens),)
