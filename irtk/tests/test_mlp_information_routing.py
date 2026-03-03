"""Tests for mlp_information_routing module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_information_routing import (
    mlp_input_output_mapping, mlp_feature_amplification,
    mlp_routing_direction_analysis, mlp_cross_position_routing,
    mlp_routing_summary,
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


def test_input_output_mapping_structure(model, tokens):
    result = mlp_input_output_mapping(model, tokens, layer=0)
    assert result["input_norm"] >= 0
    assert result["output_norm"] >= 0
    assert result["amplification"] >= 0


def test_input_output_cosine(model, tokens):
    result = mlp_input_output_mapping(model, tokens, layer=0)
    assert -1 <= result["input_output_cosine"] <= 1


def test_feature_amplification_structure(model, tokens):
    result = mlp_feature_amplification(model, tokens, layer=0, top_k=5)
    assert len(result["per_neuron"]) > 0
    assert result["mean_amplification"] >= 0


def test_feature_amplification_sorted(model, tokens):
    result = mlp_feature_amplification(model, tokens, layer=0)
    ratios = [n["amplification_ratio"] for n in result["per_neuron"]]
    assert ratios == sorted(ratios, reverse=True)


def test_routing_direction_structure(model, tokens):
    result = mlp_routing_direction_analysis(model, tokens, layer=0)
    assert len(result["directions"]) > 0
    assert result["effective_rank"] > 0


def test_routing_direction_variance(model, tokens):
    result = mlp_routing_direction_analysis(model, tokens, layer=0)
    total_var = sum(d["variance_explained"] for d in result["directions"])
    assert total_var <= 1.01  # May not sum to 1 if truncated


def test_cross_position_routing_structure(model, tokens):
    result = mlp_cross_position_routing(model, tokens, layer=0)
    assert -1 <= result["mean_similarity"] <= 1
    assert isinstance(result["is_position_specific"], bool)


def test_cross_position_routing_positions(model, tokens):
    result = mlp_cross_position_routing(model, tokens, layer=0)
    assert len(result["per_position"]) == 5


def test_routing_summary_structure(model, tokens):
    result = mlp_routing_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert p["amplification"] >= 0
