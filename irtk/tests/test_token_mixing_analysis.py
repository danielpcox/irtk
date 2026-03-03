"""Tests for token_mixing_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.token_mixing_analysis import (
    mixing_matrix,
    mixing_speed,
    information_spread,
    self_information_retention,
    mixing_entropy,
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


class TestMixingMatrix:
    def test_basic(self, model, tokens):
        result = mixing_matrix(model, tokens, layer=0)
        assert "mixing_matrix" in result
        assert "diagonal" in result
        assert "strongest_mixing_pair" in result

    def test_shape(self, model, tokens):
        result = mixing_matrix(model, tokens, layer=0)
        assert result["mixing_matrix"].shape == (len(tokens), len(tokens))


class TestMixingSpeed:
    def test_basic(self, model, tokens):
        result = mixing_speed(model, tokens)
        assert "per_layer" in result
        assert "mixing_acceleration" in result

    def test_all_layers(self, model, tokens):
        result = mixing_speed(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers


class TestInformationSpread:
    def test_basic(self, model, tokens):
        result = information_spread(model, tokens, source_pos=0, layer=0)
        assert "spread_vector" in result
        assert "reach" in result
        assert "entropy" in result

    def test_spread_shape(self, model, tokens):
        result = information_spread(model, tokens, source_pos=0, layer=0)
        assert result["spread_vector"].shape == (len(tokens),)


class TestSelfInformationRetention:
    def test_basic(self, model, tokens):
        result = self_information_retention(model, tokens)
        assert "retention_per_layer" in result
        assert "per_position_retention" in result
        assert "most_retained" in result

    def test_shapes(self, model, tokens):
        result = self_information_retention(model, tokens)
        assert result["per_position_retention"].shape == (len(tokens),)


class TestMixingEntropy:
    def test_basic(self, model, tokens):
        result = mixing_entropy(model, tokens)
        assert "per_layer" in result
        assert "entropy_trend" in result
        assert "most_uniform_layer" in result

    def test_all_layers(self, model, tokens):
        result = mixing_entropy(model, tokens)
        assert len(result["per_layer"]) == model.cfg.n_layers
