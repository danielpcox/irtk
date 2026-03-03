"""Tests for mlp_input_output_mapping module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_input_output_mapping import (
    mlp_linearity_measure, mlp_input_selectivity,
    mlp_transformation_structure, mlp_output_diversity,
    mlp_mapping_summary,
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


def test_linearity_structure(model, tokens):
    result = mlp_linearity_measure(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_linear"], bool)


def test_linearity_cosine_range(model, tokens):
    result = mlp_linearity_measure(model, tokens, layer=0)
    for p in result["per_position"]:
        assert -1.1 <= p["input_output_cosine"] <= 1.1
        assert p["input_norm"] >= 0
        assert p["output_norm"] >= 0


def test_selectivity_structure(model, tokens):
    result = mlp_input_selectivity(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert result["mean_norm"] >= 0
    assert result["coefficient_of_variation"] >= 0


def test_selectivity_flag(model, tokens):
    result = mlp_input_selectivity(model, tokens, layer=0)
    assert isinstance(result["is_selective"], bool)


def test_transformation_structure(model, tokens):
    result = mlp_transformation_structure(model, tokens, layer=0)
    assert len(result["per_position"]) == 5
    assert isinstance(result["is_amplifying"], bool)


def test_transformation_values(model, tokens):
    result = mlp_transformation_structure(model, tokens, layer=0)
    for p in result["per_position"]:
        assert p["norm_change"] >= 0
        assert -1.1 <= p["direction_change_cosine"] <= 1.1


def test_diversity_structure(model, tokens):
    result = mlp_output_diversity(model, tokens, layer=0)
    assert -1.1 <= result["mean_output_similarity"] <= 1.1
    assert isinstance(result["is_diverse"], bool)


def test_summary_structure(model, tokens):
    result = mlp_mapping_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = mlp_mapping_summary(model, tokens)
    for p in result["per_layer"]:
        assert isinstance(p["is_linear"], bool)
        assert isinstance(p["is_selective"], bool)
