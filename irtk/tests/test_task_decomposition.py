"""Tests for task_decomposition module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.task_decomposition import (
    subtask_identification,
    functional_specialization,
    task_component_alignment,
    component_cooperation_analysis,
    task_difficulty_decomposition,
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
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


@pytest.fixture
def metric_fns():
    return {
        "task_a": lambda logits: float(logits[-1, 0] - logits[-1, 1]),
        "task_b": lambda logits: float(logits[-1, 2] - logits[-1, 3]),
    }


class TestSubtaskIdentification:
    def test_basic(self, model, tokens, metric_fns):
        result = subtask_identification(model, tokens, metric_fns)
        assert "component_subtask_effects" in result
        assert "primary_subtask" in result
        assert "subtask_components" in result

    def test_all_components_present(self, model, tokens, metric_fns):
        result = subtask_identification(model, tokens, metric_fns)
        n_components = model.cfg.n_layers * model.cfg.n_heads + model.cfg.n_layers
        assert len(result["component_subtask_effects"]) == n_components

    def test_tasks_match(self, model, tokens, metric_fns):
        result = subtask_identification(model, tokens, metric_fns)
        for comp, effects in result["component_subtask_effects"].items():
            for task in metric_fns:
                assert task in effects


class TestFunctionalSpecialization:
    def test_basic(self, model, tokens, metric_fn):
        result = functional_specialization(model, tokens, metric_fn)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "specialization_scores" in result
        assert "most_specialized" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = functional_specialization(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["attn_effects"].shape == (nl, nh)
        assert result["mlp_effects"].shape == (nl,)

    def test_effects_nonneg(self, model, tokens, metric_fn):
        result = functional_specialization(model, tokens, metric_fn)
        assert np.all(result["attn_effects"] >= 0)
        assert np.all(result["mlp_effects"] >= 0)


class TestTaskComponentAlignment:
    def test_basic(self, model, tokens, metric_fns):
        result = task_component_alignment(model, tokens, metric_fns)
        assert "alignment_matrix" in result
        assert "task_overlap" in result
        assert "component_selectivity" in result

    def test_selectivity_range(self, model, tokens, metric_fns):
        result = task_component_alignment(model, tokens, metric_fns)
        for comp, sel in result["component_selectivity"].items():
            assert 0 <= sel <= 1.01


class TestComponentCooperationAnalysis:
    def test_basic(self, model, tokens, metric_fn):
        result = component_cooperation_analysis(model, tokens, metric_fn)
        assert "individual_effects" in result
        assert "pair_effects" in result
        assert "cooperation_scores" in result
        assert "mean_cooperation" in result

    def test_individual_nonneg(self, model, tokens, metric_fn):
        result = component_cooperation_analysis(model, tokens, metric_fn)
        for comp, effect in result["individual_effects"].items():
            assert effect >= 0


class TestTaskDifficultyDecomposition:
    def test_basic(self, model, tokens, metric_fn):
        tokens_hard = jnp.array([1, 2, 3, 4, 5])
        result = task_difficulty_decomposition(model, tokens, tokens_hard, metric_fn)
        assert "easy_effects" in result
        assert "hard_effects" in result
        assert "difficulty_sensitive" in result
        assert "difficulty_insensitive" in result

    def test_effects_nonneg(self, model, tokens, metric_fn):
        tokens_hard = jnp.array([1, 2, 3, 4, 5])
        result = task_difficulty_decomposition(model, tokens, tokens_hard, metric_fn)
        for comp, effect in result["easy_effects"].items():
            assert effect >= 0
        for comp, effect in result["hard_effects"].items():
            assert effect >= 0
