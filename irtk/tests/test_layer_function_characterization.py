"""Tests for layer_function_characterization module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_function_characterization import (
    layer_effect_on_logits, layer_attn_mlp_decomposition,
    layer_information_change, layer_role_classification,
    layer_characterization_summary,
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


def test_logit_effect_structure(model, tokens):
    result = layer_effect_on_logits(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_logit_effect_fields(model, tokens):
    result = layer_effect_on_logits(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["logit_mse"] >= 0
        assert isinstance(p["changes_prediction"], bool)


def test_decomposition_structure(model, tokens):
    result = layer_attn_mlp_decomposition(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_decomposition_fraction(model, tokens):
    result = layer_attn_mlp_decomposition(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert 0 <= p["attn_fraction"] <= 1


def test_information_change_structure(model, tokens):
    result = layer_information_change(model, tokens)
    assert len(result["per_layer"]) == 2


def test_information_change_values(model, tokens):
    result = layer_information_change(model, tokens)
    for p in result["per_layer"]:
        assert p["angular_change"] >= 0
        assert p["norm_change"] > 0


def test_role_classification_structure(model, tokens):
    result = layer_role_classification(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_role_classification_valid(model, tokens):
    result = layer_role_classification(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["role"] in ("refining", "redirecting", "amplifying", "maintaining")


def test_summary_structure(model, tokens):
    result = layer_characterization_summary(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "role" in p
        assert 0 <= p["attn_fraction"] <= 1
