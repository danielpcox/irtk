"""Tests for attribution graph construction and analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder
from irtk.attribution_graphs import (
    build_attribution_graph,
    node_importance,
    prune_graph,
    visualize_attribution_graph,
    attribution_graph_faithfulness,
    AttributionGraph,
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


def _make_saes(model):
    d = model.cfg.d_model
    saes = {}
    for layer in range(model.cfg.n_layers):
        hook = f"blocks.{layer}.hook_resid_post"
        key = jax.random.PRNGKey(layer + 100)
        saes[hook] = SparseAutoencoder(d, 32, key=key)
    return saes


def _metric(logits):
    return float(logits[-1, 0])


# ─── Build Attribution Graph ─────────────────────────────────────────────


class TestBuildAttributionGraph:
    def test_returns_graph(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        assert isinstance(graph, AttributionGraph)

    def test_has_nodes(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        assert len(graph.nodes) > 0

    def test_node_labels_match(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        assert len(graph.node_labels) == len(graph.nodes)

    def test_importances_shape(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        assert len(graph.node_importances) == len(graph.nodes)

    def test_empty_saes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, {}, tokens, _metric)
        assert len(graph.nodes) == 0


# ─── Node Importance ─────────────────────────────────────────────────────


class TestNodeImportance:
    def test_returns_dict(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = node_importance(graph)
        assert "importances" in result
        assert "top_nodes" in result

    def test_degree_method(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = node_importance(graph, method="degree")
        assert result["method"] == "degree"

    def test_empty_graph(self):
        graph = AttributionGraph(nodes=[], edges=[], node_labels=[], node_importances=np.array([]), n_layers=2)
        result = node_importance(graph)
        assert len(result["importances"]) == 0


# ─── Prune Graph ─────────────────────────────────────────────────────────


class TestPruneGraph:
    def test_pruned_smaller(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric, threshold=0.0001)
        pruned = prune_graph(graph, threshold=0.5)
        assert len(pruned.nodes) <= len(graph.nodes)

    def test_max_nodes(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric, threshold=0.0001)
        pruned = prune_graph(graph, max_nodes=5)
        assert len(pruned.nodes) <= 5

    def test_node_importance_method(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric, threshold=0.0001)
        pruned = prune_graph(graph, method="node_importance", threshold=0.5)
        assert len(pruned.nodes) <= len(graph.nodes)


# ─── Visualize Attribution Graph ──────────────────────────────────────────


class TestVisualizeAttributionGraph:
    def test_returns_dict(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = visualize_attribution_graph(graph, top_k_nodes=10)
        assert "nodes" in result
        assert "edges" in result
        assert "n_nodes" in result

    def test_top_k_limit(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = visualize_attribution_graph(graph, top_k_nodes=3)
        assert result["n_nodes"] <= 3

    def test_empty_graph(self):
        graph = AttributionGraph(nodes=[], edges=[], node_labels=[], node_importances=np.array([]), n_layers=2)
        result = visualize_attribution_graph(graph)
        assert result["n_nodes"] == 0


# ─── Attribution Graph Faithfulness ──────────────────────────────────────


class TestAttributionGraphFaithfulness:
    def test_returns_dict(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = attribution_graph_faithfulness(model, graph, saes, tokens, _metric)
        assert "full_metric" in result
        assert "graph_metric" in result
        assert "faithfulness" in result
        assert "feature_coverage" in result

    def test_coverage_in_range(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = attribution_graph_faithfulness(model, graph, saes, tokens, _metric)
        assert 0 <= result["feature_coverage"] <= 1.0

    def test_full_metric_is_float(self):
        model = _make_model()
        saes = _make_saes(model)
        tokens = jnp.array([0, 1, 2, 3])
        graph = build_attribution_graph(model, saes, tokens, _metric)
        result = attribution_graph_faithfulness(model, graph, saes, tokens, _metric)
        assert isinstance(result["full_metric"], float)
