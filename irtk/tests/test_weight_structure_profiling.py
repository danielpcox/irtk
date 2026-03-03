"""Tests for weight_structure_profiling module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_structure_profiling import (
    weight_spectral_profile, weight_sparsity_profile,
    weight_alignment_profile, weight_norm_distribution,
    weight_structure_summary,
)

@pytest.fixture
def model():
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
    return jax.tree.unflatten(treedef, new_leaves)

def test_spectral_structure(model):
    result = weight_spectral_profile(model, layer=0)
    assert "W_Q" in result and "W_in" in result
    assert "condition_number" in result["W_Q"]
    assert "effective_rank" in result["W_Q"]

def test_spectral_values(model):
    result = weight_spectral_profile(model, layer=0)
    assert result["W_Q"]["condition_number"] > 0
    assert result["W_Q"]["spectral_norm"] > 0
    assert 1 <= len(result["W_Q"]["top_5_singular"]) <= 5

def test_sparsity_structure(model):
    result = weight_sparsity_profile(model, layer=0)
    assert "W_Q" in result and "W_in" in result
    assert "sparsity" in result["W_Q"]

def test_sparsity_values(model):
    result = weight_sparsity_profile(model, layer=0)
    for name in ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]:
        assert 0 <= result[name]["sparsity"] <= 1.0
        assert result[name]["mean_abs_weight"] > 0

def test_alignment_structure(model):
    result = weight_alignment_profile(model, layer=0)
    assert "pairwise_alignments" in result
    assert len(result["pairwise_alignments"]) == 3  # C(3,2)

def test_alignment_values(model):
    result = weight_alignment_profile(model, layer=0)
    for pair in result["pairwise_alignments"]:
        assert -1.0 <= pair["cosine_similarity"] <= 1.0

def test_norm_distribution_structure(model):
    result = weight_norm_distribution(model)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2

def test_norm_distribution_values(model):
    result = weight_norm_distribution(model)
    for p in result["per_layer"]:
        assert p["W_Q_norm"] > 0
        assert p["total_attn_norm"] > 0
        assert p["total_mlp_norm"] > 0

def test_structure_summary(model):
    result = weight_structure_summary(model)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "W_Q_condition" in p
