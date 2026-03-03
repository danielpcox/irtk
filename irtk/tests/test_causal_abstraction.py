"""Tests for causal_abstraction module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.causal_abstraction import (
    interchange_intervention,
    causal_abstraction_test,
    distributed_alignment_search,
    multi_variable_alignment,
    abstraction_quality_score,
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


@pytest.fixture
def tokens2():
    return jnp.array([1, 6, 11, 16, 21, 26, 31, 36])


def metric_fn(logits):
    return float(logits[-1, 0])


class TestInterchangeIntervention:
    def test_output_keys(self, model, tokens, tokens2):
        r = interchange_intervention(model, tokens, tokens2, "blocks.0.hook_resid_post")
        assert "base_logits" in r
        assert "patched_logits" in r
        assert "kl_divergence" in r
        assert "logit_diff" in r

    def test_kl_nonneg(self, model, tokens, tokens2):
        r = interchange_intervention(model, tokens, tokens2, "blocks.0.hook_resid_post")
        assert r["kl_divergence"] >= -1e-5

    def test_same_input_no_change(self, model, tokens):
        r = interchange_intervention(model, tokens, tokens, "blocks.0.hook_resid_post")
        assert r["logit_diff"] < 1e-3


class TestCausalAbstractionTest:
    def test_output_keys(self, model, tokens, tokens2):
        pairs = [(tokens, tokens2, 0.1)]
        r = causal_abstraction_test(model, pairs, "blocks.0.hook_resid_post", metric_fn)
        assert "alignment_scores" in r
        assert "mean_alignment" in r
        assert "n_aligned" in r
        assert "alignment_rate" in r

    def test_alignment_rate_bounded(self, model, tokens, tokens2):
        pairs = [(tokens, tokens2, 0.1), (tokens2, tokens, -0.1)]
        r = causal_abstraction_test(model, pairs, "blocks.0.hook_resid_post", metric_fn)
        assert 0.0 <= r["alignment_rate"] <= 1.0


class TestDistributedAlignmentSearch:
    def test_output_keys(self, model, tokens, tokens2):
        r = distributed_alignment_search(model, tokens, tokens2, metric_fn, layer=1)
        assert "full_patch_effect" in r
        assert "direction_effects" in r
        assert "dims_for_half_effect" in r

    def test_shapes(self, model, tokens, tokens2):
        r = distributed_alignment_search(model, tokens, tokens2, metric_fn, layer=1)
        assert r["direction_effects"].shape == (model.cfg.d_model,)
        assert r["top_direction"].shape == (model.cfg.d_model,)


class TestMultiVariableAlignment:
    def test_output_keys(self, model, tokens, tokens2):
        hooks = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"]
        r = multi_variable_alignment(model, [tokens, tokens2], hooks, metric_fn)
        assert "hook_effects" in r
        assert "hook_rankings" in r
        assert "total_effect" in r
        assert "complementarity" in r

    def test_effects_nonneg(self, model, tokens, tokens2):
        hooks = ["blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"]
        r = multi_variable_alignment(model, [tokens], hooks, metric_fn)
        for v in r["hook_effects"].values():
            assert v >= 0


class TestAbstractionQualityScore:
    def test_output_keys(self, model, tokens, tokens2):
        r = abstraction_quality_score(model, tokens, tokens2, "blocks.0.hook_resid_post", metric_fn)
        assert "faithfulness" in r
        assert "specificity" in r
        assert "quality_score" in r
        assert "intervention_effect" in r

    def test_scores_bounded(self, model, tokens, tokens2):
        r = abstraction_quality_score(model, tokens, tokens2, "blocks.0.hook_resid_post", metric_fn)
        assert 0.0 <= r["faithfulness"] <= 1.0 + 1e-5
        assert r["quality_score"] >= 0
