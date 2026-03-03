"""Tests for mlp_nonlinearity_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_nonlinearity_analysis import (
    activation_survival_rate, nonlinearity_distortion,
    activation_magnitude_shift, neuron_selectivity_profile,
    mlp_nonlinearity_summary,
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


def test_survival_rate_structure(model, tokens):
    result = activation_survival_rate(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert 0 <= result["mean_survival_rate"] <= 1


def test_survival_rate_values(model, tokens):
    result = activation_survival_rate(model, tokens, layer=0)
    for p in result["per_position"]:
        assert 0 <= p["pre_positive_rate"] <= 1
        assert 0 <= p["post_active_rate"] <= 1


def test_distortion_structure(model, tokens):
    result = nonlinearity_distortion(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_low_distortion"], bool)


def test_distortion_cosine_range(model, tokens):
    result = nonlinearity_distortion(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.1 <= p["cosine"] <= 1.1
        assert p["norm_ratio"] >= 0


def test_magnitude_shift_structure(model, tokens):
    result = activation_magnitude_shift(model, tokens, layer=0)
    assert result["pre_mean_magnitude"] >= 0
    assert result["post_mean_magnitude"] >= 0
    assert result["magnitude_ratio"] >= 0


def test_magnitude_shift_max(model, tokens):
    result = activation_magnitude_shift(model, tokens, layer=0)
    assert result["pre_max"] >= result["pre_mean_magnitude"]
    assert result["post_max"] >= result["post_mean_magnitude"]


def test_selectivity_structure(model, tokens):
    result = neuron_selectivity_profile(model, tokens, layer=0)
    assert len(result["top_selective"]) <= 10
    assert 0 <= result["mean_selectivity"] <= 1


def test_summary_structure(model, tokens):
    result = mlp_nonlinearity_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = mlp_nonlinearity_summary(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["survival_rate"] <= 1
        assert isinstance(p["is_low_distortion"], bool)
