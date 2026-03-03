"""Tests for residual_stream_information module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_stream_information import (
    residual_prediction_entropy, information_added_per_layer,
    residual_kl_from_final, residual_mutual_information_proxy,
    residual_information_summary,
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


def test_prediction_entropy_structure(model, tokens):
    result = residual_prediction_entropy(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["final_entropy"] >= 0


def test_prediction_entropy_values(model, tokens):
    result = residual_prediction_entropy(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["entropy"] >= 0


def test_info_added_structure(model, tokens):
    result = information_added_per_layer(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert result["total_info"] >= 0


def test_info_added_values(model, tokens):
    result = information_added_per_layer(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert isinstance(p["is_informative"], bool)


def test_kl_from_final_structure(model, tokens):
    result = residual_kl_from_final(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    assert 0 <= result["convergence_layer"] < 2


def test_kl_from_final_values(model, tokens):
    result = residual_kl_from_final(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["kl_from_final"] >= 0
    # Last layer should have KL = 0
    assert result["per_layer"][-1]["kl_from_final"] < 0.01


def test_mi_proxy_structure(model, tokens):
    result = residual_mutual_information_proxy(model, tokens, layer=0)
    assert result["logit_variance"] >= 0
    assert result["logit_mean_magnitude"] >= 0


def test_summary_structure(model, tokens):
    result = residual_information_summary(model, tokens, position=-1)
    assert result["final_entropy"] >= 0
    assert result["total_info_added"] >= 0


def test_summary_convergence(model, tokens):
    result = residual_information_summary(model, tokens, position=-1)
    assert 0 <= result["convergence_layer"] < 2
    assert result["initial_kl"] >= 0
