"""Tests for mlp_activation_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_activation_analysis import (
    mlp_activation_distribution,
    dead_neuron_analysis,
    neuron_token_correlation,
    activation_sparsity_profile,
    neuron_logit_attribution,
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


class TestMlpActivationDistribution:
    def test_basic(self, model, tokens):
        result = mlp_activation_distribution(model, tokens)
        assert "mean_activation" in result
        assert "std_activation" in result
        assert "max_activation" in result
        assert "sparsity" in result
        assert "kurtosis" in result

    def test_shapes(self, model, tokens):
        result = mlp_activation_distribution(model, tokens)
        nl = model.cfg.n_layers
        assert result["mean_activation"].shape == (nl,)
        assert result["sparsity"].shape == (nl,)

    def test_nonneg(self, model, tokens):
        result = mlp_activation_distribution(model, tokens)
        assert np.all(result["mean_activation"] >= 0)
        assert np.all(result["std_activation"] >= 0)


class TestDeadNeuronAnalysis:
    def test_basic(self, model, tokens):
        result = dead_neuron_analysis(model, [tokens])
        assert "dead_fraction" in result
        assert "dead_neurons" in result
        assert "total_dead" in result
        assert "total_neurons" in result

    def test_shapes(self, model, tokens):
        result = dead_neuron_analysis(model, [tokens])
        nl = model.cfg.n_layers
        assert result["dead_fraction"].shape == (nl,)

    def test_fractions_valid(self, model, tokens):
        result = dead_neuron_analysis(model, [tokens])
        assert np.all(result["dead_fraction"] >= 0)
        assert np.all(result["dead_fraction"] <= 1.0)


class TestNeuronTokenCorrelation:
    def test_basic(self, model, tokens):
        result = neuron_token_correlation(model, tokens, layer=0)
        assert "neuron_activations" in result
        assert "top_neurons" in result
        assert "position_means" in result

    def test_top_neurons_count(self, model, tokens):
        result = neuron_token_correlation(model, tokens, layer=0, top_k=3)
        assert len(result["top_neurons"]) == 3

    def test_position_means_shape(self, model, tokens):
        result = neuron_token_correlation(model, tokens, layer=0)
        assert result["position_means"].shape == (len(tokens),)


class TestActivationSparsityProfile:
    def test_basic(self, model, tokens):
        result = activation_sparsity_profile(model, tokens)
        assert "layer_sparsity" in result
        assert "position_sparsity" in result
        assert "mean_active_neurons" in result
        assert "effective_width" in result

    def test_shapes(self, model, tokens):
        result = activation_sparsity_profile(model, tokens)
        nl = model.cfg.n_layers
        seq_len = len(tokens)
        assert result["layer_sparsity"].shape == (nl,)
        assert result["position_sparsity"].shape == (nl, seq_len)

    def test_sparsity_range(self, model, tokens):
        result = activation_sparsity_profile(model, tokens)
        assert np.all(result["layer_sparsity"] >= 0)
        assert np.all(result["layer_sparsity"] <= 1.0)


class TestNeuronLogitAttribution:
    def test_basic(self, model, tokens):
        result = neuron_logit_attribution(model, tokens, layer=0)
        assert "neuron_logit_effects" in result
        assert "top_neuron_token_pairs" in result
        assert "neuron_total_effects" in result

    def test_shapes(self, model, tokens):
        result = neuron_logit_attribution(model, tokens, layer=0)
        d_vocab = model.cfg.d_vocab
        assert result["neuron_logit_effects"].shape[1] == d_vocab

    def test_total_nonneg(self, model, tokens):
        result = neuron_logit_attribution(model, tokens, layer=0)
        assert np.all(result["neuron_total_effects"] >= 0)
