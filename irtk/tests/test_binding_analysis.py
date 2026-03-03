"""Tests for entity-attribute binding analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.binding_analysis import (
    entity_attribute_binding,
    binding_attention_pattern,
    cross_position_binding_score,
    binding_through_layers,
    multi_entity_disambiguation,
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


# ─── Entity Attribute Binding ────────────────────────────────────────────────


class TestEntityAttributeBinding:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = entity_attribute_binding(model, tokens, entity_pos=3, attribute_pos=2)
        assert "binding_strength" in result
        assert "attention_flow" in result
        assert "residual_similarity" in result
        assert "layer_binding" in result

    def test_layer_binding_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = entity_attribute_binding(model, tokens, entity_pos=3, attribute_pos=2)
        assert len(result["layer_binding"]) == 2  # n_layers=2

    def test_specific_layer(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = entity_attribute_binding(
            model, tokens, entity_pos=3, attribute_pos=2, layer=0
        )
        assert len(result["layer_binding"]) == 1

    def test_similarity_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = entity_attribute_binding(model, tokens, entity_pos=3, attribute_pos=2)
        assert -1.01 <= result["residual_similarity"] <= 1.01


# ─── Binding Attention Pattern ───────────────────────────────────────────────


class TestBindingAttentionPattern:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_attention_pattern(model, tokens, entity_pos=3, attribute_pos=2)
        assert "head_scores" in result
        assert "top_binding_heads" in result
        assert "max_score" in result

    def test_head_scores_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_attention_pattern(model, tokens, entity_pos=3, attribute_pos=2)
        assert result["head_scores"].shape == (2, 4)

    def test_scores_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_attention_pattern(model, tokens, entity_pos=3, attribute_pos=2)
        assert np.all(result["head_scores"] >= 0)


# ─── Cross Position Binding Score ────────────────────────────────────────────


class TestCrossPositionBindingScore:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_position_binding_score(model, tokens)
        assert "binding_matrix" in result
        assert "strongest_pair" in result
        assert "mean_binding" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_position_binding_score(model, tokens)
        assert result["binding_matrix"].shape == (4, 4)

    def test_diagonal_is_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cross_position_binding_score(model, tokens)
        np.testing.assert_allclose(
            np.diag(result["binding_matrix"]), 1.0, atol=0.01
        )


# ─── Binding Through Layers ─────────────────────────────────────────────────


class TestBindingThroughLayers:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_through_layers(model, tokens, entity_pos=3, attribute_pos=2)
        assert "layer_similarities" in result
        assert "peak_layer" in result
        assert "binding_emerges" in result
        assert "binding_trend" in result

    def test_similarities_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_through_layers(model, tokens, entity_pos=3, attribute_pos=2)
        assert len(result["layer_similarities"]) == 2

    def test_peak_layer_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_through_layers(model, tokens, entity_pos=3, attribute_pos=2)
        assert 0 <= result["peak_layer"] < 2

    def test_trend_is_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = binding_through_layers(model, tokens, entity_pos=3, attribute_pos=2)
        assert result["binding_trend"] in ("increasing", "decreasing", "non_monotonic", "flat")


# ─── Multi Entity Disambiguation ────────────────────────────────────────────


class TestMultiEntityDisambiguation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = multi_entity_disambiguation(
            model, tokens, entity_positions=[1, 4], query_pos=5
        )
        assert "entity_similarities" in result
        assert "query_to_entity_attention" in result
        assert "discrimination_score" in result

    def test_with_metric(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = multi_entity_disambiguation(
            model, tokens, entity_positions=[1, 4], query_pos=5,
            metric_fn=_metric,
        )
        assert len(result["ablation_effects"]) == 2

    def test_single_entity(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = multi_entity_disambiguation(
            model, tokens, entity_positions=[1], query_pos=3
        )
        assert result["discrimination_score"] == 0.0
