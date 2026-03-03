"""Tests for mlp_specialization_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_specialization_analysis import (
    mlp_input_selectivity, mlp_output_targeting,
    mlp_cross_layer_division, mlp_activation_sparsity_profile,
    mlp_specialization_summary,
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


def test_input_selectivity_structure(model, tokens):
    result = mlp_input_selectivity(model, tokens, layer=0, top_k=5)
    assert len(result["per_neuron"]) > 0
    assert isinstance(result["is_selective"], bool)


def test_input_selectivity_range(model, tokens):
    result = mlp_input_selectivity(model, tokens, layer=0)
    for n in result["per_neuron"]:
        assert 0 <= n["selectivity"] <= 1


def test_output_targeting_structure(model, tokens):
    result = mlp_output_targeting(model, tokens, layer=0, top_k=5)
    assert len(result["per_neuron"]) > 0
    assert result["mean_logit_range"] >= 0


def test_output_targeting_logits(model, tokens):
    result = mlp_output_targeting(model, tokens, layer=0)
    for n in result["per_neuron"]:
        assert n["top_promoted_logit"] >= n["top_suppressed_logit"]


def test_cross_layer_division_structure(model, tokens):
    result = mlp_cross_layer_division(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_specialized"], bool)


def test_cross_layer_similarity(model, tokens):
    result = mlp_cross_layer_division(model, tokens)
    for s in result["cross_layer_similarities"]:
        assert -1 <= s["cosine"] <= 1


def test_sparsity_profile_structure(model, tokens):
    result = mlp_activation_sparsity_profile(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["sparsity_trend"] in ("increasing", "decreasing")


def test_sparsity_profile_range(model, tokens):
    result = mlp_activation_sparsity_profile(model, tokens)
    for p in result["per_layer"]:
        assert 0 <= p["mean_sparsity"] <= 1
        assert 0 <= p["active_neuron_overlap"] <= 1


def test_specialization_summary_structure(model, tokens):
    result = mlp_specialization_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["most_impactful_layer"] < 2
    for p in result["per_layer"]:
        assert p["output_norm"] >= 0
        assert p["logit_impact"] >= 0
