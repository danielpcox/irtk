"""Tests for attention_weight_structure module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_weight_structure import (
    qk_weight_norms, ov_weight_norms,
    weight_matrix_rank_analysis, qk_ov_alignment,
    weight_structure_summary,
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


def test_qk_norms_structure(model):
    result = qk_weight_norms(model)
    assert len(result["per_layer"]) == 2


def test_qk_norms_positive(model):
    result = qk_weight_norms(model)
    for p in result["per_layer"]:
        assert p["mean_q_norm"] > 0
        assert p["mean_k_norm"] > 0


def test_ov_norms_structure(model):
    result = ov_weight_norms(model)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert len(p["per_head"]) == 4


def test_ov_norms_positive(model):
    result = ov_weight_norms(model)
    for p in result["per_layer"]:
        for h in p["per_head"]:
            assert h["v_norm"] > 0
            assert h["o_norm"] > 0


def test_rank_analysis_structure(model):
    result = weight_matrix_rank_analysis(model, layer=0, head=0)
    assert "W_Q" in result["matrices"]
    assert "W_K" in result["matrices"]
    assert "W_V" in result["matrices"]
    assert "W_O" in result["matrices"]


def test_rank_analysis_values(model):
    result = weight_matrix_rank_analysis(model, layer=0, head=0)
    for name, data in result["matrices"].items():
        assert data["effective_rank"] > 0
        assert data["max_sv"] >= data["min_sv"]


def test_qk_ov_alignment_range(model):
    result = qk_ov_alignment(model, layer=0, head=0)
    assert -1.1 <= result["qk_ov_alignment"] <= 1.1
    assert isinstance(result["is_aligned"], bool)


def test_summary_structure(model):
    result = weight_structure_summary(model)
    assert len(result["per_layer"]) == 2


def test_summary_norms(model):
    result = weight_structure_summary(model)
    for p in result["per_layer"]:
        assert p["mean_q_norm"] > 0
        assert p["mean_v_norm"] > 0
