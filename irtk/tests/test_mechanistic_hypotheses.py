"""Tests for mechanistic hypothesis formation and validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.mechanistic_hypotheses import (
    propose_hypotheses,
    validate_hypothesis,
    hypothesis_to_circuit,
    compose_hypotheses,
    explain_prediction,
    MechanisticHypothesis,
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


# ─── Propose Hypotheses ──────────────────────────────────────────────────


class TestProposeHypotheses:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = propose_hypotheses(model, seqs)
        assert "hypotheses" in result
        assert "n_hypotheses" in result
        assert "component_coverage" in result

    def test_hypotheses_are_objects(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = propose_hypotheses(model, seqs)
        for h in result["hypotheses"]:
            assert isinstance(h, MechanisticHypothesis)
            assert h.component != ""
            assert h.role != ""

    def test_coverage_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = propose_hypotheses(model, seqs)
        assert 0 <= result["component_coverage"] <= 1.0

    def test_multiple_sequences(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = propose_hypotheses(model, seqs)
        assert result["n_hypotheses"] >= 0


# ─── Validate Hypothesis ─────────────────────────────────────────────────


class TestValidateHypothesis:
    def test_returns_dict(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L0H0", role="previous_token",
            description="test", hook_name="blocks.0.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = validate_hypothesis(model, h, seqs)
        assert "passes" in result
        assert "consistency_score" in result
        assert "evidence" in result

    def test_consistency_in_range(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L0H0", role="current_token",
            description="test", hook_name="blocks.0.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = validate_hypothesis(model, h, seqs)
        assert 0 <= result["consistency_score"] <= 1.0

    def test_with_metric(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L0H0", role="previous_token",
            description="test", hook_name="blocks.0.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = validate_hypothesis(model, h, seqs, metric_fn=_metric)
        assert isinstance(result["ablation_effect"], float)


# ─── Hypothesis to Circuit ────────────────────────────────────────────────


class TestHypothesisToCircuit:
    def test_returns_dict(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L1H2", role="induction",
            description="test", hook_name="blocks.1.attn.hook_pattern"
        )
        result = hypothesis_to_circuit(h, model)
        assert "nodes" in result
        assert "edges" in result
        assert "component_roles" in result

    def test_has_nodes(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L1H2", role="induction",
            description="test", hook_name="blocks.1.attn.hook_pattern"
        )
        result = hypothesis_to_circuit(h, model)
        assert len(result["nodes"]) >= 1
        assert "L1H2" in result["nodes"]

    def test_edges_connect_nodes(self):
        model = _make_model()
        h = MechanisticHypothesis(
            component="L1H2", role="test",
            description="test", hook_name="blocks.1.attn.hook_pattern"
        )
        result = hypothesis_to_circuit(h, model)
        for src, tgt in result["edges"]:
            assert src in result["nodes"]
            assert tgt in result["nodes"]


# ─── Compose Hypotheses ──────────────────────────────────────────────────


class TestComposeHypotheses:
    def test_returns_dict(self):
        model = _make_model()
        h1 = MechanisticHypothesis(
            component="L0H0", role="previous_token",
            description="test", hook_name="blocks.0.attn.hook_pattern"
        )
        h2 = MechanisticHypothesis(
            component="L1H0", role="induction",
            description="test", hook_name="blocks.1.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = compose_hypotheses(h1, h2, model, seqs)
        assert "composes" in result
        assert "interaction_score" in result
        assert "relationship" in result

    def test_relationship_valid(self):
        model = _make_model()
        h1 = MechanisticHypothesis(
            component="L0H0", role="prev", description="test",
            hook_name="blocks.0.attn.hook_pattern"
        )
        h2 = MechanisticHypothesis(
            component="L1H0", role="ind", description="test",
            hook_name="blocks.1.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3])]
        result = compose_hypotheses(h1, h2, model, seqs)
        assert result["relationship"] in ("sequential", "parallel", "independent")

    def test_interaction_non_negative(self):
        model = _make_model()
        h1 = MechanisticHypothesis(
            component="L0H0", role="prev", description="test",
            hook_name="blocks.0.attn.hook_pattern"
        )
        h2 = MechanisticHypothesis(
            component="L0H1", role="curr", description="test",
            hook_name="blocks.0.attn.hook_pattern"
        )
        seqs = [jnp.array([0, 1, 2, 3])]
        result = compose_hypotheses(h1, h2, model, seqs)
        assert result["interaction_score"] >= 0


# ─── Explain Prediction ──────────────────────────────────────────────────


class TestExplainPrediction:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        hyps = propose_hypotheses(model, seqs)["hypotheses"][:3]
        if not hyps:
            return  # skip if no hypotheses generated
        tokens = jnp.array([0, 1, 2, 3])
        result = explain_prediction(model, tokens, hyps, _metric)
        assert "full_metric" in result
        assert "component_effects" in result
        assert "explanation_order" in result

    def test_effects_match_hypotheses(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        hyps = propose_hypotheses(model, seqs)["hypotheses"][:2]
        if not hyps:
            return
        tokens = jnp.array([0, 1, 2, 3])
        result = explain_prediction(model, tokens, hyps, _metric)
        assert len(result["component_effects"]) == len(hyps)

    def test_total_explained_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        hyps = propose_hypotheses(model, seqs)["hypotheses"][:2]
        if not hyps:
            return
        tokens = jnp.array([0, 1, 2, 3])
        result = explain_prediction(model, tokens, hyps, _metric)
        assert result["total_explained"] >= 0
