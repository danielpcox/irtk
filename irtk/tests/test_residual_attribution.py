"""Tests for residual_attribution module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_attribution import (
    per_component_residual_contribution,
    cumulative_residual_buildup,
    directional_attribution,
    component_interference_analysis,
    residual_decomposition_at_unembed,
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


class TestPerComponentResidualContribution:
    def test_basic(self, model, tokens):
        result = per_component_residual_contribution(model, tokens)
        assert "embed_contribution" in result
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "contribution_norms" in result
        assert "dominant_component" in result

    def test_shapes(self, model, tokens):
        result = per_component_residual_contribution(model, tokens)
        d = model.cfg.d_model
        nl = model.cfg.n_layers
        assert result["embed_contribution"].shape == (d,)
        assert result["attn_contributions"].shape == (nl, d)
        assert result["mlp_contributions"].shape == (nl, d)

    def test_norms_match(self, model, tokens):
        result = per_component_residual_contribution(model, tokens)
        embed_norm = float(np.linalg.norm(result["embed_contribution"]))
        np.testing.assert_allclose(result["contribution_norms"]["embed"], embed_norm, atol=1e-4)


class TestCumulativeResidualBuildup:
    def test_basic(self, model, tokens):
        result = cumulative_residual_buildup(model, tokens)
        assert "residual_norms" in result
        assert "residual_directions" in result
        assert "direction_changes" in result
        assert "growth_rates" in result

    def test_shapes(self, model, tokens):
        result = cumulative_residual_buildup(model, tokens)
        nl = model.cfg.n_layers
        d = model.cfg.d_model
        assert result["residual_norms"].shape == (nl + 1,)
        assert result["residual_directions"].shape == (nl + 1, d)
        assert result["direction_changes"].shape == (nl,)
        assert result["growth_rates"].shape == (nl,)

    def test_angles_nonnegative(self, model, tokens):
        result = cumulative_residual_buildup(model, tokens)
        assert np.all(result["direction_changes"] >= 0)


class TestDirectionalAttribution:
    def test_basic(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = directional_attribution(model, tokens, direction)
        assert "embed_attribution" in result
        assert "attn_attributions" in result
        assert "mlp_attributions" in result
        assert "total_attribution" in result
        assert "attribution_fractions" in result

    def test_shapes(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = directional_attribution(model, tokens, direction)
        nl = model.cfg.n_layers
        assert result["attn_attributions"].shape == (nl,)
        assert result["mlp_attributions"].shape == (nl,)

    def test_total_is_sum(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = directional_attribution(model, tokens, direction)
        expected = result["embed_attribution"] + float(np.sum(result["attn_attributions"])) + float(np.sum(result["mlp_attributions"]))
        np.testing.assert_allclose(result["total_attribution"], expected, atol=1e-4)


class TestComponentInterferenceAnalysis:
    def test_basic(self, model, tokens):
        result = component_interference_analysis(model, tokens)
        assert "pairwise_alignment" in result
        assert "constructive_pairs" in result
        assert "destructive_pairs" in result
        assert "net_alignment" in result

    def test_alignment_range(self, model, tokens):
        result = component_interference_analysis(model, tokens)
        for pair, cos in result["pairwise_alignment"].items():
            assert -1.01 <= cos <= 1.01

    def test_pairs_consistent(self, model, tokens):
        result = component_interference_analysis(model, tokens)
        for pair in result["constructive_pairs"]:
            assert result["pairwise_alignment"][pair] > 0.1
        for pair in result["destructive_pairs"]:
            assert result["pairwise_alignment"][pair] < -0.1


class TestResidualDecompositionAtUnembed:
    def test_basic(self, model, tokens):
        result = residual_decomposition_at_unembed(model, tokens)
        assert "component_logits" in result
        assert "top_tokens_per_component" in result
        assert "total_logits" in result

    def test_has_all_components(self, model, tokens):
        result = residual_decomposition_at_unembed(model, tokens)
        nl = model.cfg.n_layers
        assert "embed" in result["component_logits"]
        for l in range(nl):
            assert f"attn_L{l}" in result["component_logits"]
            assert f"mlp_L{l}" in result["component_logits"]

    def test_total_logits_shape(self, model, tokens):
        result = residual_decomposition_at_unembed(model, tokens)
        assert result["total_logits"].shape == (model.cfg.d_vocab,)

    def test_top_tokens_format(self, model, tokens):
        result = residual_decomposition_at_unembed(model, tokens)
        for comp, top_list in result["top_tokens_per_component"].items():
            assert len(top_list) == 5
            for tok_idx, logit_val in top_list:
                assert isinstance(tok_idx, int)
                assert isinstance(logit_val, float)
