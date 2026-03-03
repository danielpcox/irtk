"""Tests for model internal consistency."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.model_internal_consistency import (
    logit_lens_consistency, residual_norm_monotonicity,
    component_orthogonality, output_sensitivity,
    model_consistency_summary,
)


@pytest.fixture
def model_and_tokens():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    model = jax.tree.unflatten(treedef, new_leaves)
    tokens = jnp.array([1, 5, 10, 15, 20])
    return model, tokens


def test_logit_lens_consistency_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_consistency(model, tokens, position=-1)
    assert "per_layer" in result
    assert "final_prediction" in result
    assert len(result["per_layer"]) == 2


def test_logit_lens_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_consistency(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert -1.0 <= p["logit_cosine"] <= 1.0
    assert result["per_layer"][-1]["agrees_with_final"] is True


def test_norm_monotonicity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_norm_monotonicity(model, tokens, position=-1)
    assert "per_layer" in result
    assert "is_monotonic" in result
    assert "growth_factor" in result


def test_norm_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_norm_monotonicity(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["norm"] > 0
    assert result["growth_factor"] > 0


def test_orthogonality_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_orthogonality(model, tokens, layer=0, position=-1)
    assert "cosine" in result
    assert "orthogonality" in result
    assert "is_orthogonal" in result


def test_orthogonality_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_orthogonality(model, tokens, layer=0, position=-1)
    assert -1.0 <= result["cosine"] <= 1.0
    assert 0 <= result["orthogonality"] <= 1.0


def test_sensitivity_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = output_sensitivity(model, tokens, position=-1)
    assert "sensitivity" in result
    assert "base_logit_norm" in result


def test_sensitivity_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = output_sensitivity(model, tokens, position=-1)
    assert result["sensitivity"] >= 0
    assert result["base_logit_norm"] > 0


def test_consistency_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = model_consistency_summary(model, tokens, position=-1)
    assert "logit_lens_agreement" in result
    assert "norm_monotonic" in result
    assert "mean_orthogonality" in result
    assert 0 <= result["logit_lens_agreement"] <= 1.0
