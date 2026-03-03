"""Tests for head composition analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_composition_analysis import (
    qk_composition_scores, ov_composition_scores,
    strongest_compositions, composition_path_strength,
    head_composition_summary,
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


def test_qk_composition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_composition_scores(model, source_layer=0, dest_layer=1)
    assert "scores" in result
    assert "per_pair" in result
    assert result["scores"].shape == (4, 4)


def test_qk_composition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = qk_composition_scores(model, source_layer=0, dest_layer=1)
    for p in result["per_pair"]:
        assert p["composition_score"] >= 0


def test_ov_composition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_composition_scores(model, source_layer=0, dest_layer=1)
    assert result["scores"].shape == (4, 4)
    assert len(result["per_pair"]) == 16


def test_ov_composition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_composition_scores(model, source_layer=0, dest_layer=1)
    for p in result["per_pair"]:
        assert p["composition_score"] >= 0


def test_strongest_compositions(model_and_tokens):
    model, tokens = model_and_tokens
    result = strongest_compositions(model, source_layer=0, dest_layer=1, top_k=3)
    assert "qk_top" in result
    assert "ov_top" in result
    assert len(result["qk_top"]) == 3


def test_strongest_descending(model_and_tokens):
    model, tokens = model_and_tokens
    result = strongest_compositions(model, source_layer=0, dest_layer=1, top_k=3)
    qk_scores = [p["composition_score"] for p in result["qk_top"]]
    assert qk_scores == sorted(qk_scores, reverse=True)


def test_path_strength_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = composition_path_strength(model, tokens, source_layer=0, source_head=0, dest_layer=1, dest_head=0)
    assert "source_output_norm" in result
    assert "composition_strength" in result


def test_path_strength_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = composition_path_strength(model, tokens, source_layer=0, source_head=0, dest_layer=1, dest_head=0)
    assert result["source_output_norm"] >= 0
    assert result["composition_strength"] >= 0


def test_composition_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_composition_summary(model, tokens)
    assert "per_layer_pair" in result
    assert len(result["per_layer_pair"]) == 1  # 2 layers -> 1 pair
