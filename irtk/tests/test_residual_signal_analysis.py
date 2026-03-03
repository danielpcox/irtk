"""Tests for residual_signal_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_signal_analysis import (
    signal_noise_ratio, component_interference,
    residual_coherence, update_orthogonality,
    residual_signal_summary,
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


def test_snr_structure(model, tokens):
    result = signal_noise_ratio(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["final_snr"] >= 0


def test_snr_values(model, tokens):
    result = signal_noise_ratio(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["signal"] >= 0
        assert p["noise"] >= 0
        assert 0 <= p["signal_fraction"] <= 1.01


def test_interference_structure(model, tokens):
    result = component_interference(model, tokens, layer=0, position=-1)
    assert -1.1 <= result["cosine"] <= 1.1
    assert isinstance(result["has_interference"], bool)


def test_interference_norms(model, tokens):
    result = component_interference(model, tokens, layer=0, position=-1)
    assert result["attn_norm"] >= 0
    assert result["mlp_norm"] >= 0
    assert result["combined_norm"] >= 0


def test_coherence_structure(model, tokens):
    result = residual_coherence(model, tokens, position=-1)
    assert len(result["per_pair"]) == 1  # 2 layers -> 1 pair
    assert isinstance(result["is_coherent"], bool)


def test_coherence_cosine_range(model, tokens):
    result = residual_coherence(model, tokens, position=-1)
    for p in result["per_pair"]:
        assert -1.1 <= p["cosine"] <= 1.1


def test_orthogonality_structure(model, tokens):
    result = update_orthogonality(model, tokens, position=-1)
    assert 0 <= result["mean_orthogonality"] <= 1.01
    assert result["mean_abs_cosine"] >= 0


def test_summary_structure(model, tokens):
    result = residual_signal_summary(model, tokens, position=-1)
    assert result["final_snr"] >= 0
    assert isinstance(result["is_coherent"], bool)


def test_summary_orthogonality(model, tokens):
    result = residual_signal_summary(model, tokens, position=-1)
    assert 0 <= result["mean_orthogonality"] <= 1.01
