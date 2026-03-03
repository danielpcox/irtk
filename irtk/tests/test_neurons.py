"""Tests for neuron analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.neurons import (
    get_neuron_activations,
    neuron_activation_stats,
    top_activating_tokens,
    neuron_to_logit,
    neuron_logit_effects,
    dead_neuron_fraction,
    neuron_attribution,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestGetNeuronActivations:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        acts = get_neuron_activations(cache, layer=0)
        assert acts.shape == (4, model.cfg.d_mlp)

    def test_pre_vs_post(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        pre = get_neuron_activations(cache, layer=0, hook="pre")
        post = get_neuron_activations(cache, layer=0, hook="post")
        assert pre.shape == post.shape


class TestNeuronActivationStats:
    def test_keys(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        stats = neuron_activation_stats(model, seqs, layer=0)
        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "firing_rate" in stats
        assert "l1_norm" in stats

    def test_shapes(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        stats = neuron_activation_stats(model, seqs, layer=0)
        d_mlp = model.cfg.d_mlp
        assert stats["mean"].shape == (d_mlp,)
        assert stats["std"].shape == (d_mlp,)
        assert stats["firing_rate"].shape == (d_mlp,)

    def test_firing_rate_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        stats = neuron_activation_stats(model, seqs, layer=0)
        assert np.all(stats["firing_rate"] >= 0)
        assert np.all(stats["firing_rate"] <= 1)


class TestTopActivatingTokens:
    def test_returns_k_items(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        top = top_activating_tokens(model, seqs, layer=0, neuron=0, k=3)
        assert len(top) == 3

    def test_sorted_descending(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        top = top_activating_tokens(model, seqs, layer=0, neuron=0, k=4)
        activations = [a for _, _, a in top]
        assert activations == sorted(activations, reverse=True)


class TestNeuronToLogit:
    def test_returns_promoted_suppressed(self):
        model = _make_model()
        promoted, suppressed = neuron_to_logit(model, layer=0, neuron=0, k=5)
        assert len(promoted) == 5
        assert len(suppressed) == 5

    def test_promoted_have_higher_values(self):
        model = _make_model()
        promoted, suppressed = neuron_to_logit(model, layer=0, neuron=0, k=3)
        if promoted and suppressed:
            assert promoted[0][1] >= suppressed[0][1]


class TestNeuronLogitEffects:
    def test_shape(self):
        model = _make_model()
        effects = neuron_logit_effects(model, layer=0)
        assert effects.shape == (model.cfg.d_mlp, model.cfg.d_vocab)


class TestDeadNeuronFraction:
    def test_returns_fraction_and_mask(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        frac, is_dead = dead_neuron_fraction(model, seqs, layer=0)
        assert 0.0 <= frac <= 1.0
        assert is_dead.shape == (model.cfg.d_mlp,)
        assert is_dead.dtype == bool

    def test_empty_dataset(self):
        model = _make_model()
        frac, is_dead = dead_neuron_fraction(model, [], layer=0)
        assert frac == 0.0


class TestNeuronAttribution:
    def test_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        attr = neuron_attribution(model, cache, layer=0, token=5)
        assert attr.shape == (model.cfg.d_mlp,)
