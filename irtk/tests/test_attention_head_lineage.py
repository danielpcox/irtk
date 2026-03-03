"""Tests for attention_head_lineage module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_lineage import (
    head_to_head_attention, head_output_influence,
    head_composition_chain, layer_head_dependency_graph,
    attention_head_lineage_summary,
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

def test_head_to_head_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_to_head_attention(model, tokens, src_layer=0, dst_layer=1)
    assert "interaction_matrix" in result
    assert result["interaction_matrix"].shape == (4, 4)
    assert "strongest_connections" in result

def test_head_to_head_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_to_head_attention(model, tokens, src_layer=0, dst_layer=1)
    matrix = result["interaction_matrix"]
    assert float(jnp.sum(matrix)) > 0
    for conn in result["strongest_connections"]:
        assert conn["strength"] >= 0

def test_output_influence_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_influence(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4

def test_output_influence_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_influence(model, tokens, layer=0)
    for h in result["per_head"]:
        assert h["q_influence"] >= 0
        assert h["k_influence"] >= 0
        assert h["v_influence"] >= 0
        assert h["dominant_path"] in ("Q", "K", "V")

def test_composition_chain_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_composition_chain(model, tokens, start_layer=0, start_head=0)
    assert "chain" in result
    assert len(result["chain"]) == 2  # 2 layers
    assert result["chain"][0]["layer"] == 0
    assert result["chain"][0]["head"] == 0

def test_composition_chain_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_composition_chain(model, tokens, start_layer=0, start_head=0)
    assert result["chain_strength"] > 0
    assert len(result["per_step_strength"]) == 1

def test_dependency_graph_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_head_dependency_graph(model, tokens)
    assert "edges" in result
    assert len(result["edges"]) > 0

def test_dependency_graph_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = layer_head_dependency_graph(model, tokens)
    for edge in result["edges"]:
        assert edge["src_layer"] < edge["dst_layer"]
        assert edge["strength"] >= 0

def test_lineage_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_head_lineage_summary(model, tokens)
    assert "strongest_chains" in result
    assert "n_strong_edges" in result
    assert "mean_edge_strength" in result
