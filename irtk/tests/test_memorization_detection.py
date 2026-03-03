"""Tests for memorization detection analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.memorization_detection import (
    memorization_score,
    extractability_by_layer,
    generalization_gap_profile,
    memorized_token_localization,
    content_extraction_risk,
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


class TestMemorizationScore:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorization_score(model, tokens)
        assert "clean_confidence" in result
        assert "memorization_score" in result
        assert "is_likely_memorized" in result

    def test_confidence_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorization_score(model, tokens)
        assert result["clean_confidence"] > 0

    def test_score_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorization_score(model, tokens)
        assert result["memorization_score"] >= 0


class TestExtractabilityByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extractability_by_layer(model, tokens, target_token=5)
        assert "layer_ranks" in result
        assert "layer_probs" in result
        assert "extraction_layer" in result
        assert "accessibility_curve" in result

    def test_ranks_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extractability_by_layer(model, tokens, target_token=5)
        assert len(result["layer_ranks"]) == 2

    def test_probs_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extractability_by_layer(model, tokens, target_token=5)
        assert np.all(result["layer_probs"] >= 0)
        assert np.all(result["layer_probs"] <= 1.0)


class TestGeneralizationGapProfile:
    def test_returns_dict(self):
        model = _make_model()
        train_tokens = jnp.array([0, 1, 2, 3])
        test_tokens = jnp.array([4, 5, 6, 7])
        result = generalization_gap_profile(model, train_tokens, test_tokens)
        assert "train_entropies" in result
        assert "test_entropies" in result
        assert "entropy_gap" in result
        assert "representation_distance" in result

    def test_entropies_length(self):
        model = _make_model()
        result = generalization_gap_profile(model, jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]))
        assert len(result["train_entropies"]) == 2


class TestMemorizedTokenLocalization:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorized_token_localization(model, tokens, _metric)
        assert "position_importance" in result
        assert "trigger_positions" in result
        assert "most_critical_position" in result

    def test_importance_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorized_token_localization(model, tokens, _metric)
        assert len(result["position_importance"]) == 4

    def test_importance_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = memorized_token_localization(model, tokens, _metric)
        assert np.all(result["position_importance"] >= 0)


class TestContentExtractionRisk:
    def test_returns_dict(self):
        model = _make_model()
        prefix = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        result = content_extraction_risk(model, prefix, target)
        assert "per_token_probs" in result
        assert "mean_probability" in result
        assert "extraction_risk" in result
        assert "weakest_link" in result

    def test_probs_in_range(self):
        model = _make_model()
        prefix = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        result = content_extraction_risk(model, prefix, target)
        assert np.all(result["per_token_probs"] >= 0)
        assert np.all(result["per_token_probs"] <= 1.0)

    def test_risk_level_valid(self):
        model = _make_model()
        prefix = jnp.array([0, 1, 2, 3])
        target = jnp.array([4, 5])
        result = content_extraction_risk(model, prefix, target)
        assert result["extraction_risk"] in ("low", "medium", "high")
