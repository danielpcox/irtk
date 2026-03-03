"""Tests for computation_graph module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.computation_graph import (
    component_dependency_graph,
    dataflow_analysis,
    computation_cost_profile,
    critical_path_analysis,
    component_interaction_strength,
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


@pytest.fixture
def metric_fn():
    def fn(logits):
        return float(logits[-1, 0] - logits[-1, 1])
    return fn


class TestComponentDependencyGraph:
    def test_basic(self, model, tokens, metric_fn):
        result = component_dependency_graph(model, tokens, metric_fn)
        assert "edges" in result
        assert "n_edges" in result
        assert "component_names" in result
        assert "in_degree" in result
        assert "out_degree" in result

    def test_component_names(self, model, tokens, metric_fn):
        result = component_dependency_graph(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        expected = n_layers * n_heads + n_layers  # heads + MLPs
        assert len(result["component_names"]) == expected

    def test_edge_format(self, model, tokens, metric_fn):
        result = component_dependency_graph(model, tokens, metric_fn)
        for edge in result["edges"]:
            assert len(edge) == 3
            src, tgt, weight = edge
            assert isinstance(src, str)
            assert isinstance(tgt, str)
            assert isinstance(weight, float)

    def test_n_edges_matches(self, model, tokens, metric_fn):
        result = component_dependency_graph(model, tokens, metric_fn)
        assert result["n_edges"] == len(result["edges"])


class TestDataflowAnalysis:
    def test_basic(self, model, tokens):
        result = dataflow_analysis(model, tokens)
        assert "attn_throughput" in result
        assert "mlp_throughput" in result
        assert "residual_norms" in result
        assert "attn_fraction" in result
        assert "mlp_fraction" in result

    def test_shapes(self, model, tokens):
        result = dataflow_analysis(model, tokens)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        assert result["attn_throughput"].shape == (n_layers, n_heads)
        assert result["mlp_throughput"].shape == (n_layers,)
        assert result["residual_norms"].shape == (n_layers + 1,)
        assert result["attn_fraction"].shape == (n_layers,)
        assert result["mlp_fraction"].shape == (n_layers,)

    def test_fractions_sum(self, model, tokens):
        result = dataflow_analysis(model, tokens)
        totals = result["attn_fraction"] + result["mlp_fraction"]
        # Should sum to approximately 1 per layer
        np.testing.assert_allclose(totals, 1.0, atol=0.01)

    def test_throughput_nonnegative(self, model, tokens):
        result = dataflow_analysis(model, tokens)
        assert np.all(result["attn_throughput"] >= 0)
        assert np.all(result["mlp_throughput"] >= 0)
        assert np.all(result["residual_norms"] >= 0)


class TestComputationCostProfile:
    def test_basic(self, model, tokens, metric_fn):
        result = computation_cost_profile(model, tokens, metric_fn)
        assert "attn_params" in result
        assert "mlp_params" in result
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "cost_effectiveness" in result
        assert "most_cost_effective" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = computation_cost_profile(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        assert result["attn_params"].shape == (n_layers,)
        assert result["mlp_params"].shape == (n_layers,)
        assert result["attn_effects"].shape == (n_layers,)
        assert result["mlp_effects"].shape == (n_layers,)

    def test_cost_effectiveness_keys(self, model, tokens, metric_fn):
        result = computation_cost_profile(model, tokens, metric_fn)
        n_layers = model.cfg.n_layers
        for l in range(n_layers):
            assert f"attn_L{l}" in result["cost_effectiveness"]
            assert f"mlp_L{l}" in result["cost_effectiveness"]

    def test_effects_nonnegative(self, model, tokens, metric_fn):
        result = computation_cost_profile(model, tokens, metric_fn)
        assert np.all(result["attn_effects"] >= 0)
        assert np.all(result["mlp_effects"] >= 0)


class TestCriticalPathAnalysis:
    def test_basic(self, model, tokens, metric_fn):
        result = critical_path_analysis(model, tokens, metric_fn)
        assert "critical_path" in result
        assert "path_effects" in result
        assert "total_path_effect" in result
        assert "path_length" in result

    def test_path_length(self, model, tokens, metric_fn):
        result = critical_path_analysis(model, tokens, metric_fn)
        assert result["path_length"] == model.cfg.n_layers
        assert len(result["critical_path"]) == model.cfg.n_layers
        assert len(result["path_effects"]) == model.cfg.n_layers

    def test_total_path_effect(self, model, tokens, metric_fn):
        result = critical_path_analysis(model, tokens, metric_fn)
        np.testing.assert_allclose(
            result["total_path_effect"],
            sum(result["path_effects"]),
            atol=1e-6,
        )

    def test_path_components_valid(self, model, tokens, metric_fn):
        result = critical_path_analysis(model, tokens, metric_fn)
        for comp in result["critical_path"]:
            assert comp.startswith("attn_") or comp.startswith("mlp_")


class TestComponentInteractionStrength:
    def test_basic(self, model, tokens, metric_fn):
        result = component_interaction_strength(model, tokens, metric_fn, layer_a=0, layer_b=1)
        assert "interaction_matrix" in result
        assert "strongest_interaction" in result
        assert "mean_interaction" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = component_interaction_strength(model, tokens, metric_fn)
        n_heads = model.cfg.n_heads
        n_comp = n_heads + 1  # heads + MLP
        assert result["interaction_matrix"].shape == (n_comp, n_comp)

    def test_strongest_interaction_valid(self, model, tokens, metric_fn):
        result = component_interaction_strength(model, tokens, metric_fn)
        src, tgt = result["strongest_interaction"]
        assert isinstance(src, str)
        assert isinstance(tgt, str)

    def test_interactions_nonnegative(self, model, tokens, metric_fn):
        result = component_interaction_strength(model, tokens, metric_fn)
        assert np.all(result["interaction_matrix"] >= 0)
