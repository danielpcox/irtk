"""Tests for layer ablation effects."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_ablation_effects import (
    layer_zero_ablation, component_ablation,
    mean_ablation, cumulative_ablation,
    layer_ablation_summary,
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


def test_zero_ablation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_zero_ablation(model, tokens, position=-1)
    assert "per_layer" in result
    assert "base_prediction" in result
    assert len(result["per_layer"]) == 2


def test_zero_ablation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_zero_ablation(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert p["kl_divergence"] >= 0


def test_component_ablation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_ablation(model, tokens, layer=0, position=-1)
    assert "attn_kl" in result
    assert "mlp_kl" in result
    assert "more_important" in result


def test_component_ablation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_ablation(model, tokens, layer=0, position=-1)
    assert result["attn_kl"] >= 0
    assert result["mlp_kl"] >= 0
    assert result["more_important"] in ("attn", "mlp")


def test_mean_ablation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_ablation(model, tokens, layer=0, position=-1)
    assert "kl_divergence" in result
    assert "prediction_changed" in result


def test_mean_ablation_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_ablation(model, tokens, layer=0, position=-1)
    assert result["kl_divergence"] >= 0


def test_cumulative_ablation_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_ablation(model, tokens, position=-1)
    assert "per_n_layers_removed" in result
    assert len(result["per_n_layers_removed"]) == 2


def test_cumulative_ablation_monotonic(model_and_tokens):
    model, tokens = model_and_tokens
    result = cumulative_ablation(model, tokens, position=-1)
    kls = [p["kl_divergence"] for p in result["per_n_layers_removed"]]
    # More layers removed should generally increase KL (or at least not decrease much)
    assert kls[-1] >= kls[0] - 0.1  # last >= first (with tolerance)


def test_ablation_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_ablation_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "kl_divergence" in p
        assert "more_important" in p
