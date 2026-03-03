"""Tests for position-centric attribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.token_position_attribution import (
    position_gradient_attribution,
    position_flow_through_layers,
    position_interaction_matrix,
    position_specific_ablation,
    position_to_logit_attribution,
)


def _make_model(seed=42):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(seed)
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


# ─── Position Gradient Attribution ────────────────────────────────────────────


class TestPositionGradientAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_gradient_attribution(model, tokens, _metric)
        assert "position_scores" in result
        assert "most_important_position" in result
        assert "attribution_entropy" in result

    def test_scores_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_gradient_attribution(model, tokens, _metric)
        assert len(result["position_scores"]) == 4

    def test_scores_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_gradient_attribution(model, tokens, _metric)
        assert np.all(result["position_scores"] >= 0)

    def test_position_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_gradient_attribution(model, tokens, _metric)
        assert 0 <= result["most_important_position"] < 4


# ─── Position Flow Through Layers ─────────────────────────────────────────────


class TestPositionFlowThroughLayers:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_flow_through_layers(model, tokens, source_pos=0)
        assert "attention_flow" in result
        assert "flow_persistence" in result
        assert "decay_rate" in result

    def test_flow_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_flow_through_layers(model, tokens, source_pos=0)
        assert result["attention_flow"].shape == (2, 4)

    def test_persistence_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_flow_through_layers(model, tokens, source_pos=0)
        assert len(result["flow_persistence"]) == 2


# ─── Position Interaction Matrix ──────────────────────────────────────────────


class TestPositionInteractionMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_interaction_matrix(model, tokens, _metric)
        assert "interaction_matrix" in result
        assert "strongest_interaction" in result
        assert "individual_effects" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_interaction_matrix(model, tokens, _metric)
        assert result["interaction_matrix"].shape == (4, 4)

    def test_symmetric(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_interaction_matrix(model, tokens, _metric)
        np.testing.assert_allclose(
            result["interaction_matrix"],
            result["interaction_matrix"].T,
            atol=1e-6,
        )


# ─── Position Specific Ablation ──────────────────────────────────────────────


class TestPositionSpecificAblation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_specific_ablation(model, tokens, _metric)
        assert "ablation_effects" in result
        assert "most_critical_position" in result
        assert "effect_variance" in result

    def test_effects_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_specific_ablation(model, tokens, _metric)
        assert len(result["ablation_effects"]) == 4

    def test_subset_positions(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_specific_ablation(model, tokens, _metric, positions=[0, 2])
        assert len(result["ablation_effects"]) == 2


# ─── Position to Logit Attribution ────────────────────────────────────────────


class TestPositionToLogitAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_to_logit_attribution(model, tokens, target_token=0)
        assert "position_attributions" in result
        assert "most_contributing" in result
        assert "normalized_attributions" in result

    def test_attributions_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_to_logit_attribution(model, tokens, target_token=0)
        assert len(result["position_attributions"]) == 4

    def test_normalized_sums_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = position_to_logit_attribution(model, tokens, target_token=0)
        total = np.sum(result["normalized_attributions"])
        if total > 0:
            np.testing.assert_allclose(total, 1.0, atol=0.01)
