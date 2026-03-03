"""Tests for head OV eigenanalysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_ov_eigenanalysis import (
    ov_eigenspectrum, ov_copying_score,
    ov_unembed_projection, ov_cross_head_alignment,
    ov_eigenanalysis_summary,
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


def test_ov_eigenspectrum_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_eigenspectrum(model, layer=0, head=0)
    assert "singular_values" in result
    assert "effective_rank" in result
    assert "top_sv" in result
    assert "sv_concentration" in result


def test_ov_eigenspectrum_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_eigenspectrum(model, layer=0, head=0)
    assert result["effective_rank"] >= 1.0
    assert result["top_sv"] > 0
    assert 0 <= result["sv_concentration"] <= 1.0


def test_ov_copying_score_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_copying_score(model, layer=0, head=0)
    assert "trace" in result
    assert "frobenius_norm" in result
    assert "identity_score" in result
    assert "is_copying" in result


def test_ov_copying_score_range(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_copying_score(model, layer=0, head=0)
    assert result["frobenius_norm"] >= 0
    assert -1.0 <= result["identity_score"] <= 1.0


def test_ov_unembed_projection_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_unembed_projection(model, layer=0, head=0, top_k=5)
    assert "top_affected_tokens" in result
    assert len(result["top_affected_tokens"]) == 5
    assert "mean_effect" in result


def test_ov_unembed_projection_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_unembed_projection(model, layer=0, head=0)
    assert result["max_effect"] >= result["mean_effect"]


def test_ov_cross_head_alignment_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_cross_head_alignment(model, layer=0)
    assert "pairs" in result
    assert "n_aligned" in result
    assert len(result["pairs"]) == 6  # C(4,2) = 6


def test_ov_cross_head_alignment_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_cross_head_alignment(model, layer=0)
    for p in result["pairs"]:
        assert -1.0 <= p["cosine"] <= 1.0


def test_ov_eigenanalysis_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = ov_eigenanalysis_summary(model)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "mean_rank" in p
        assert "mean_copy_score" in p
