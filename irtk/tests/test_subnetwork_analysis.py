"""Tests for subnetwork_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.subnetwork_analysis import (
    extract_important_components,
    subnetwork_faithfulness,
    subnetwork_minimality,
    compare_subnetworks,
    greedy_subnetwork_search,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


def metric_fn(logits):
    return float(logits[-1, 0])


class TestExtractImportantComponents:
    def test_output_keys(self, model, tokens):
        r = extract_important_components(model, tokens, metric_fn, threshold=0.1)
        assert "important_heads" in r
        assert "important_mlps" in r
        assert "ablation_effects" in r
        assert "n_important" in r
        assert "fraction_important" in r

    def test_effects_nonneg(self, model, tokens):
        r = extract_important_components(model, tokens, metric_fn, threshold=0.0)
        for val in r["ablation_effects"].values():
            assert val >= 0.0

    def test_fraction_bounded(self, model, tokens):
        r = extract_important_components(model, tokens, metric_fn, threshold=0.1)
        assert 0.0 <= r["fraction_important"] <= 1.0

    def test_all_components_present(self, model, tokens):
        r = extract_important_components(model, tokens, metric_fn, threshold=0.0)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        # Should have effects for all heads and MLPs
        expected_count = n_layers * n_heads + n_layers
        assert len(r["ablation_effects"]) == expected_count


class TestSubnetworkFaithfulness:
    def test_output_keys(self, model, tokens):
        r = subnetwork_faithfulness(model, tokens, metric_fn, heads=[(0, 0)], mlps=[0])
        assert "full_metric" in r
        assert "subnetwork_metric" in r
        assert "faithfulness" in r
        assert "absolute_error" in r
        assert "relative_error" in r

    def test_full_subnetwork_faithful(self, model, tokens):
        # Including all components should be very faithful
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        all_mlps = list(range(n_layers))
        r = subnetwork_faithfulness(model, tokens, metric_fn, heads=all_heads, mlps=all_mlps)
        assert abs(r["absolute_error"]) < 1e-3

    def test_empty_subnetwork(self, model, tokens):
        r = subnetwork_faithfulness(model, tokens, metric_fn, heads=[], mlps=[])
        # Empty subnetwork zeroes everything - should differ from full
        assert r["absolute_error"] > 0 or abs(r["full_metric"]) < 1e-5


class TestSubnetworkMinimality:
    def test_output_keys(self, model, tokens):
        r = subnetwork_minimality(model, tokens, metric_fn,
                                   heads=[(0, 0), (1, 0)], mlps=[0])
        assert "removable_heads" in r
        assert "removable_mlps" in r
        assert "n_removable" in r
        assert "is_minimal" in r
        assert "component_necessity" in r

    def test_single_component(self, model, tokens):
        r = subnetwork_minimality(model, tokens, metric_fn,
                                   heads=[(0, 0)], mlps=[])
        # With single component, it's either necessary or not
        assert isinstance(r["is_minimal"], (bool, np.bool_))

    def test_consistency(self, model, tokens):
        heads = [(0, 0), (0, 1)]
        mlps = [0]
        r = subnetwork_minimality(model, tokens, metric_fn, heads=heads, mlps=mlps)
        # n_removable should equal sum of removable heads + mlps
        assert r["n_removable"] == len(r["removable_heads"]) + len(r["removable_mlps"])


class TestCompareSubnetworks:
    def test_output_keys(self, model, tokens):
        sub_a = {"heads": [(0, 0), (0, 1)], "mlps": [0]}
        sub_b = {"heads": [(1, 0), (1, 1)], "mlps": [1]}
        r = compare_subnetworks(model, tokens, metric_fn, sub_a, sub_b)
        assert "faithfulness_a" in r
        assert "faithfulness_b" in r
        assert "overlap_heads" in r
        assert "overlap_mlps" in r
        assert "jaccard_similarity" in r

    def test_identical_subnetworks(self, model, tokens):
        sub = {"heads": [(0, 0)], "mlps": [0]}
        r = compare_subnetworks(model, tokens, metric_fn, sub, sub)
        assert r["jaccard_similarity"] == 1.0

    def test_disjoint_subnetworks(self, model, tokens):
        sub_a = {"heads": [(0, 0)], "mlps": [0]}
        sub_b = {"heads": [(1, 1)], "mlps": [1]}
        r = compare_subnetworks(model, tokens, metric_fn, sub_a, sub_b)
        assert r["jaccard_similarity"] == 0.0
        assert len(r["overlap_heads"]) == 0
        assert len(r["overlap_mlps"]) == 0


class TestGreedySubnetworkSearch:
    def test_output_keys(self, model, tokens):
        r = greedy_subnetwork_search(model, tokens, metric_fn, target_faithfulness=0.5)
        assert "selected_heads" in r
        assert "selected_mlps" in r
        assert "faithfulness_trajectory" in r
        assert "n_components_needed" in r
        assert "final_faithfulness" in r

    def test_trajectory_monotonic_ish(self, model, tokens):
        r = greedy_subnetwork_search(model, tokens, metric_fn, target_faithfulness=0.99)
        # Trajectory should generally increase (not strictly due to ablation effects)
        traj = r["faithfulness_trajectory"]
        assert len(traj) > 0

    def test_components_count(self, model, tokens):
        r = greedy_subnetwork_search(model, tokens, metric_fn, target_faithfulness=0.5)
        assert r["n_components_needed"] == len(r["selected_heads"]) + len(r["selected_mlps"])
