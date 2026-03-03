"""Tests for weight_probing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_probing import (
    spectral_signatures,
    role_specialization,
    weight_activation_alignment,
    pruning_sensitivity,
    weight_geometry,
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
    return jnp.array([0, 5, 10, 15, 20])


class TestSpectralSignatures:
    def test_attn(self, model):
        result = spectral_signatures(model, layer=0, component="attn")
        assert "W_Q" in result
        assert "W_K" in result
        assert "W_V" in result
        assert "W_O" in result

    def test_mlp(self, model):
        result = spectral_signatures(model, layer=0, component="mlp")
        assert "W_in" in result
        assert "W_out" in result

    def test_rank_positive(self, model):
        result = spectral_signatures(model, layer=0)
        assert result["W_Q"]["effective_rank"] > 0


class TestRoleSpecialization:
    def test_all_heads(self, model):
        result = role_specialization(model, layer=0)
        assert "specialization_scores" in result
        assert "similarity_matrix" in result
        assert "most_unique_head" in result

    def test_single_head(self, model):
        result = role_specialization(model, layer=0, head=0)
        assert "qk_norm" in result
        assert "ov_norm" in result

    def test_specialization_range(self, model):
        result = role_specialization(model, layer=0)
        for s in result["specialization_scores"]:
            assert -0.5 <= float(s) <= 1.5


class TestWeightActivationAlignment:
    def test_basic(self, model, tokens):
        result = weight_activation_alignment(model, tokens, layer=0, head=0)
        assert "q_alignment" in result
        assert "v_alignment" in result

    def test_alignment_nonneg(self, model, tokens):
        result = weight_activation_alignment(model, tokens, layer=0, head=0)
        assert result["q_alignment"] >= 0


class TestPruningSensitivity:
    def test_attn(self, model):
        result = pruning_sensitivity(model, layer=0, component="attn")
        assert "importance_scores" in result
        assert "pruning_order" in result
        assert "cumulative_norm_loss" in result

    def test_mlp(self, model):
        result = pruning_sensitivity(model, layer=0, component="mlp")
        assert "importance_scores" in result

    def test_cumulative_bounded(self, model):
        result = pruning_sensitivity(model, layer=0, component="attn")
        assert float(result["cumulative_norm_loss"][-1]) <= 1.01


class TestWeightGeometry:
    def test_basic(self, model):
        result = weight_geometry(model)
        assert "norm_profile" in result
        assert "isotropy_profile" in result
        assert "inter_layer_similarity" in result

    def test_profile_length(self, model):
        result = weight_geometry(model)
        assert len(result["norm_profile"]) == model.cfg.n_layers
        assert result["inter_layer_similarity"].shape == (model.cfg.n_layers, model.cfg.n_layers)
