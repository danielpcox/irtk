"""Tests for mlp_output_space module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_output_space import (
    mlp_output_rank, mlp_output_direction_analysis,
    mlp_output_token_alignment, mlp_output_position_variation,
    mlp_output_summary,
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


def test_output_rank_structure(model, tokens):
    result = mlp_output_rank(model, tokens, layer=0)
    assert result["effective_rank"] > 0
    assert result["dim_for_90_pct"] > 0


def test_output_rank_sv(model, tokens):
    result = mlp_output_rank(model, tokens, layer=0)
    assert result["top_singular_value"] > 0


def test_direction_analysis_structure(model, tokens):
    result = mlp_output_direction_analysis(model, tokens, layer=0, top_k=3)
    assert len(result["directions"]) > 0
    assert result["top_1_variance"] > 0


def test_direction_variance(model, tokens):
    result = mlp_output_direction_analysis(model, tokens, layer=0)
    total = sum(d["variance_explained"] for d in result["directions"])
    assert total <= 1.01


def test_token_alignment_structure(model, tokens):
    result = mlp_output_token_alignment(model, tokens, layer=0, top_k=5)
    assert len(result["top_tokens"]) == 5
    assert result["logit_range"] >= 0


def test_position_variation_structure(model, tokens):
    result = mlp_output_position_variation(model, tokens, layer=0)
    assert -1 <= result["mean_position_similarity"] <= 1
    assert isinstance(result["is_position_sensitive"], bool)


def test_position_variation_cv(model, tokens):
    result = mlp_output_position_variation(model, tokens, layer=0)
    assert result["norm_coefficient_of_variation"] >= 0


def test_summary_structure(model, tokens):
    result = mlp_output_summary(model, tokens)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = mlp_output_summary(model, tokens)
    for p in result["per_layer"]:
        assert p["effective_rank"] > 0
        assert isinstance(p["is_position_sensitive"], bool)
