"""Tests for automated circuit discovery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.circuit_discovery import (
    edge_attribution_matrix,
    iterative_circuit_pruning,
    subnetwork_probing,
    path_attribution,
    discover_circuit,
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


def _metric(logits):
    return float(logits[-1, 0])


# ─── Edge Attribution Matrix ─────────────────────────────────────────────────


class TestEdgeAttributionMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edge_attribution_matrix(model, tokens, _metric)
        assert "edge_scores" in result
        assert "top_edges" in result
        assert "sparsity" in result

    def test_matrix_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edge_attribution_matrix(model, tokens, _metric)
        assert result["edge_scores"].shape == (3, 3)  # n_layers + 1

    def test_sparsity_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = edge_attribution_matrix(model, tokens, _metric)
        assert 0 <= result["sparsity"] <= 1.0


# ─── Iterative Circuit Pruning ────────────────────────────────────────────────


class TestIterativeCircuitPruning:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = iterative_circuit_pruning(model, tokens, _metric)
        assert "circuit_components" in result
        assert "circuit_size" in result
        assert "compression_ratio" in result

    def test_compression_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = iterative_circuit_pruning(model, tokens, _metric)
        assert 0 <= result["compression_ratio"] <= 1.0

    def test_max_components_limits(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = iterative_circuit_pruning(model, tokens, _metric, max_components=3)
        assert result["circuit_size"] <= 3


# ─── Subnetwork Probing ──────────────────────────────────────────────────────


class TestSubnetworkProbing:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subnetwork_probing(model, seqs, _metric, n_random_subsets=5)
        assert "component_frequencies" in result
        assert "mean_subset_metric" in result
        assert "critical_components" in result

    def test_frequencies_are_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = subnetwork_probing(model, seqs, _metric, n_random_subsets=5)
        assert isinstance(result["component_frequencies"], dict)


# ─── Path Attribution ─────────────────────────────────────────────────────────


class TestPathAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_attribution(model, tokens, _metric, source_layer=0)
        assert "path_scores" in result
        assert "direct_effect" in result
        assert "indirect_effect" in result

    def test_direct_effect_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = path_attribution(model, tokens, _metric, source_layer=0)
        assert result["direct_effect"] >= 0


# ─── Discover Circuit ─────────────────────────────────────────────────────────


class TestDiscoverCircuit:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = discover_circuit(model, seqs, _metric)
        assert "nodes" in result
        assert "edges" in result
        assert "node_importance" in result
        assert "circuit_size" in result

    def test_edges_connect_nodes(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = discover_circuit(model, seqs, _metric)
        for src, tgt in result["edges"]:
            assert src in result["nodes"]
            assert tgt in result["nodes"]

    def test_empty_sequences(self):
        model = _make_model()
        result = discover_circuit(model, [], _metric)
        assert result["circuit_size"] == 0
