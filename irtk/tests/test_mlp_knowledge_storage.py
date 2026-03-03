"""Tests for MLP knowledge storage."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_knowledge_storage import (
    neuron_logit_effect, mlp_layer_logit_effect,
    knowledge_localization, mlp_association_structure,
    mlp_knowledge_summary,
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


def test_neuron_logit_effect_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_effect(model, layer=0, top_k=3)
    assert "per_neuron" in result
    assert len(result["per_neuron"]) == 3


def test_neuron_logit_effect_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = neuron_logit_effect(model, layer=0, top_k=3)
    for p in result["per_neuron"]:
        assert len(p["promoted"]) == 3
        assert len(p["suppressed"]) == 3
        assert p["effect_range"] > 0


def test_layer_logit_effect_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_logit_effect(model, tokens, layer=0, position=-1, top_k=3)
    assert "promoted" in result
    assert "suppressed" in result
    assert len(result["promoted"]) == 3


def test_layer_logit_effect_ordering(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_layer_logit_effect(model, tokens, layer=0, position=-1, top_k=3)
    logits = [l for _, l in result["promoted"]]
    assert logits == sorted(logits, reverse=True)


def test_knowledge_localization_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = knowledge_localization(model, tokens, target_token=5, position=-1)
    assert "per_layer" in result
    assert "most_important_layer" in result
    assert len(result["per_layer"]) == 2


def test_knowledge_localization_fractions(model_and_tokens):
    model, tokens = model_and_tokens
    result = knowledge_localization(model, tokens, target_token=5, position=-1)
    total_frac = sum(p["fraction"] for p in result["per_layer"])
    assert abs(total_frac - 1.0) < 0.01


def test_association_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_association_structure(model, layer=0, top_k=3)
    assert "effective_rank" in result
    assert "top_singular_values" in result
    assert "condition_number" in result


def test_association_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_association_structure(model, layer=0, top_k=3)
    assert result["effective_rank"] > 0
    assert len(result["top_singular_values"]) == 3


def test_knowledge_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = mlp_knowledge_summary(model, tokens, position=-1)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert "effect_norm" in p
        assert "effective_rank" in p
