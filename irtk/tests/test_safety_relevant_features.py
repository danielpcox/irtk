"""Tests for safety_relevant_features module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.safety_relevant_features import (
    refusal_direction_analysis,
    knowledge_localization,
    deception_detection_signatures,
    alignment_circuit_analysis,
    safety_feature_monitoring,
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


class TestRefusalDirectionAnalysis:
    def test_basic(self, model, tokens):
        compliant = [tokens, jnp.array([1, 2, 3, 4, 5])]
        refused = [jnp.array([10, 20, 30, 40, 49]), jnp.array([5, 15, 25, 35, 45])]
        result = refusal_direction_analysis(model, compliant, refused)
        assert "refusal_direction" in result
        assert "separation_score" in result
        assert "compliant_projections" in result
        assert "refused_projections" in result

    def test_direction_shape(self, model, tokens):
        compliant = [tokens]
        refused = [jnp.array([10, 20, 30, 40, 49])]
        result = refusal_direction_analysis(model, compliant, refused)
        assert result["refusal_direction"].shape == (model.cfg.d_model,)

    def test_separation_nonneg(self, model, tokens):
        compliant = [tokens]
        refused = [jnp.array([10, 20, 30, 40, 49])]
        result = refusal_direction_analysis(model, compliant, refused)
        assert result["separation_score"] >= 0


class TestKnowledgeLocalization:
    def test_basic(self, model, tokens, metric_fn):
        result = knowledge_localization(model, tokens, [0, 1], metric_fn)
        assert "layer_importance" in result
        assert "component_importance" in result
        assert "critical_layers" in result
        assert "attn_vs_mlp" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = knowledge_localization(model, tokens, [0], metric_fn)
        nl = model.cfg.n_layers
        assert result["layer_importance"].shape == (nl,)
        assert result["attn_vs_mlp"].shape == (nl,)

    def test_importance_nonneg(self, model, tokens, metric_fn):
        result = knowledge_localization(model, tokens, [0], metric_fn)
        assert np.all(result["layer_importance"] >= 0)

    def test_attn_vs_mlp_range(self, model, tokens, metric_fn):
        result = knowledge_localization(model, tokens, [0], metric_fn)
        assert np.all(result["attn_vs_mlp"] >= 0)
        assert np.all(result["attn_vs_mlp"] <= 1)


class TestDeceptionDetectionSignatures:
    def test_basic(self, model, tokens):
        honest = tokens
        deceptive = jnp.array([10, 20, 30, 40, 49])
        result = deception_detection_signatures(model, honest, deceptive)
        assert "layer_divergence" in result
        assert "cosine_similarity" in result
        assert "divergence_onset_layer" in result
        assert "attention_divergence" in result

    def test_shapes(self, model, tokens):
        honest = tokens
        deceptive = jnp.array([10, 20, 30, 40, 49])
        result = deception_detection_signatures(model, honest, deceptive)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["layer_divergence"].shape == (nl,)
        assert result["cosine_similarity"].shape == (nl,)
        assert result["attention_divergence"].shape == (nl, nh)

    def test_divergence_nonneg(self, model, tokens):
        honest = tokens
        deceptive = jnp.array([10, 20, 30, 40, 49])
        result = deception_detection_signatures(model, honest, deceptive)
        assert np.all(result["layer_divergence"] >= 0)


class TestAlignmentCircuitAnalysis:
    def test_basic(self, model, tokens, metric_fn):
        safety_fn = lambda logits: float(logits[-1, 2] - logits[-1, 3])
        result = alignment_circuit_analysis(model, tokens, metric_fn, safety_fn)
        assert "behavior_importance" in result
        assert "safety_importance" in result
        assert "overlap_score" in result
        assert "conflict_heads" in result
        assert "synergy_heads" in result

    def test_shapes(self, model, tokens, metric_fn):
        safety_fn = lambda logits: float(logits[-1, 2])
        result = alignment_circuit_analysis(model, tokens, metric_fn, safety_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["behavior_importance"].shape == (nl, nh)
        assert result["safety_importance"].shape == (nl, nh)

    def test_correlation_range(self, model, tokens, metric_fn):
        safety_fn = lambda logits: float(logits[-1, 2])
        result = alignment_circuit_analysis(model, tokens, metric_fn, safety_fn)
        assert -1.01 <= result["overlap_score"] <= 1.01


class TestSafetyFeatureMonitoring:
    def test_basic(self, model, tokens):
        result = safety_feature_monitoring(model, tokens)
        assert "activation_norm" in result
        assert "top_active_dimensions" in result
        assert "position_variance" in result
        assert "anomaly_score" in result

    def test_with_directions(self, model, tokens):
        d_model = model.cfg.d_model
        directions = {
            "test_dir": np.random.randn(d_model),
            "another_dir": np.random.randn(d_model),
        }
        result = safety_feature_monitoring(model, tokens, reference_directions=directions)
        assert len(result["projections"]) == 2
        assert "test_dir" in result["projections"]

    def test_norm_positive(self, model, tokens):
        result = safety_feature_monitoring(model, tokens)
        assert result["activation_norm"] >= 0

    def test_anomaly_nonneg(self, model, tokens):
        result = safety_feature_monitoring(model, tokens)
        assert result["anomaly_score"] >= 0

    def test_variance_shape(self, model, tokens):
        result = safety_feature_monitoring(model, tokens)
        assert result["position_variance"].shape == (model.cfg.d_model,)
