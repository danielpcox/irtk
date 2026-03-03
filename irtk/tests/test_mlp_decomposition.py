"""Tests for MLP decomposition analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.mlp_decomposition import (
    neuron_contribution_decompose,
    mlp_feature_directions,
    mlp_input_output_alignment,
    mlp_knowledge_storage,
    mlp_nonlinearity_analysis,
)


def _make_model(seed=42):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(seed)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestNeuronContributionDecompose:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = neuron_contribution_decompose(model, tokens, layer=0)
        assert "top_neurons" in result
        assert "neuron_contributions" in result
        assert "total_contribution" in result
        assert "top_k_fraction" in result

    def test_top_k_limit(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = neuron_contribution_decompose(model, tokens, layer=0, top_k=5)
        assert len(result["top_neurons"]) == 5

    def test_fraction_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = neuron_contribution_decompose(model, tokens, layer=0)
        assert 0.0 <= result["top_k_fraction"] <= 1.0


class TestMlpFeatureDirections:
    def test_returns_dict(self):
        model = _make_model()
        result = mlp_feature_directions(model, layer=0, top_k=5)
        assert "input_directions" in result
        assert "output_directions" in result
        assert "singular_values" in result
        assert "effective_rank" in result

    def test_directions_shape(self):
        model = _make_model()
        result = mlp_feature_directions(model, layer=0, top_k=5)
        assert result["input_directions"].shape == (5, 16)
        assert result["output_directions"].shape == (5, 16)

    def test_effective_rank_positive(self):
        model = _make_model()
        result = mlp_feature_directions(model, layer=0)
        assert result["effective_rank"] > 0


class TestMlpInputOutputAlignment:
    def test_returns_dict(self):
        model = _make_model()
        result = mlp_input_output_alignment(model, layer=0)
        assert "per_neuron_alignment" in result
        assert "mean_alignment" in result
        assert "amplifying_neurons" in result
        assert "transforming_neurons" in result

    def test_alignment_length(self):
        model = _make_model()
        result = mlp_input_output_alignment(model, layer=0)
        assert len(result["per_neuron_alignment"]) == 64  # d_mlp = 4 * d_model

    def test_alignment_in_range(self):
        model = _make_model()
        result = mlp_input_output_alignment(model, layer=0)
        assert np.all(result["per_neuron_alignment"] >= -1.01)
        assert np.all(result["per_neuron_alignment"] <= 1.01)


class TestMlpKnowledgeStorage:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mlp_knowledge_storage(model, tokens, layer=0, target_token=5)
        assert "knowledge_neurons" in result
        assert "neuron_logit_effects" in result
        assert "total_promotion" in result
        assert "knowledge_concentration" in result

    def test_knowledge_neurons_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mlp_knowledge_storage(model, tokens, layer=0, target_token=5, top_k=5)
        assert len(result["knowledge_neurons"]) == 5


class TestMlpNonlinearityAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mlp_nonlinearity_analysis(model, tokens, layer=0)
        assert "active_fraction" in result
        assert "pre_activation_stats" in result
        assert "gating_sharpness" in result
        assert "dead_neurons" in result

    def test_active_fraction_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mlp_nonlinearity_analysis(model, tokens, layer=0)
        assert 0.0 <= result["active_fraction"] <= 1.0

    def test_dead_neurons_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mlp_nonlinearity_analysis(model, tokens, layer=0)
        assert result["dead_neurons"] >= 0
