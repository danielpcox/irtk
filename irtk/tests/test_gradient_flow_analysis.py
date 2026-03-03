"""Tests for gradient_flow_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.gradient_flow_analysis import (
    gradient_norm_by_layer, gradient_component_attribution,
    gradient_saturation_analysis, gradient_bottleneck_detection,
    gradient_flow_summary,
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
    return jnp.array([1, 5, 10, 15, 20])


def test_gradient_norm_structure(model, tokens):
    result = gradient_norm_by_layer(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_stable"], bool)


def test_gradient_norm_positive(model, tokens):
    result = gradient_norm_by_layer(model, tokens)
    for p in result["per_layer"]:
        assert p["activation_norm"] > 0
        assert p["update_norm"] >= 0


def test_component_attribution_structure(model, tokens):
    result = gradient_component_attribution(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["dominant_component"] in ("attention", "mlp")


def test_component_attribution_fractions(model, tokens):
    result = gradient_component_attribution(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["attn_fraction"] <= 1
        assert 0 <= p["mlp_fraction"] <= 1
        assert abs(p["attn_fraction"] + p["mlp_fraction"] - 1.0) < 0.01


def test_saturation_structure(model, tokens):
    result = gradient_saturation_analysis(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["any_saturated"], bool)


def test_saturation_range(model, tokens):
    result = gradient_saturation_analysis(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["attention_saturation"] <= 1
        assert 0 <= p["mlp_near_zero_fraction"] <= 1


def test_bottleneck_structure(model, tokens):
    result = gradient_bottleneck_detection(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["any_bottleneck"], bool)


def test_bottleneck_rank(model, tokens):
    result = gradient_bottleneck_detection(model, tokens)
    for p in result["per_layer"]:
        assert p["update_effective_rank"] >= 0
        assert 0 <= p["top_sv_fraction"] <= 1


def test_flow_summary_structure(model, tokens):
    result = gradient_flow_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_stable"], bool)
    assert result["dominant_component"] in ("attention", "mlp")
    assert isinstance(result["any_saturated"], bool)
