"""Tests for residual_stream_convergence module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_convergence import (
    layer_to_layer_convergence, final_representation_stability,
    position_convergence, norm_convergence,
    convergence_summary,
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


def test_l2l_convergence_structure(model, tokens):
    result = layer_to_layer_convergence(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_converging"], bool)


def test_l2l_convergence_positive(model, tokens):
    result = layer_to_layer_convergence(model, tokens)
    for c in result["per_layer"]:
        assert c["absolute_change"] >= 0
        assert c["relative_change"] >= 0


def test_final_stability_structure(model, tokens):
    result = final_representation_stability(model, tokens)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["stabilization_layer"] < 2


def test_final_stability_cosines(model, tokens):
    result = final_representation_stability(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["cosine_to_final"] <= 1
    # Last layer should have cosine 1.0 to itself
    assert result["per_layer"][-1]["cosine_to_final"] > 0.99


def test_position_convergence_structure(model, tokens):
    result = position_convergence(model, tokens)
    assert len(result["per_layer"]) == 2
    assert result["position_trend"] in ("converging", "diverging", "stable")


def test_position_convergence_range(model, tokens):
    result = position_convergence(model, tokens)
    for p in result["per_layer"]:
        assert -1 <= p["mean_pairwise_similarity"] <= 1


def test_norm_convergence_structure(model, tokens):
    result = norm_convergence(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_stable"], bool)


def test_norm_convergence_positive(model, tokens):
    result = norm_convergence(model, tokens)
    for p in result["per_layer"]:
        assert p["mean_norm"] >= 0


def test_summary_structure(model, tokens):
    result = convergence_summary(model, tokens)
    assert len(result["per_layer"]) == 2
    assert isinstance(result["is_converging"], bool)
    assert isinstance(result["is_stable"], bool)
