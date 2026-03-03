"""Tests for mlp_knowledge_editing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.mlp_knowledge_editing import (
    locate_fact_in_mlps,
    rank_one_mlp_edit,
    verify_edit_effect,
    edit_side_effects,
    mlp_fact_extraction,
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


class TestLocateFactInMlps:
    def test_basic(self, model, tokens, metric_fn):
        result = locate_fact_in_mlps(model, tokens, metric_fn)
        assert "mlp_effects" in result
        assert "decisive_layer" in result
        assert "top_layers" in result
        assert "fact_distributed" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = locate_fact_in_mlps(model, tokens, metric_fn)
        assert result["mlp_effects"].shape == (model.cfg.n_layers,)

    def test_decisive_valid(self, model, tokens, metric_fn):
        result = locate_fact_in_mlps(model, tokens, metric_fn)
        assert 0 <= result["decisive_layer"] < model.cfg.n_layers


class TestRankOneMlpEdit:
    def test_basic(self, model):
        d_mlp = model.cfg.d_model * 4
        key_vec = np.random.randn(d_mlp)
        val_vec = np.random.randn(model.cfg.d_model)
        result = rank_one_mlp_edit(model, 0, key_vec, val_vec)
        assert "edited_model" in result
        assert "edit_norm" in result
        assert result["edit_norm"] > 0

    def test_edit_changes_output(self, model, tokens):
        d_mlp = model.cfg.d_model * 4
        key_vec = np.random.randn(d_mlp) * 10
        val_vec = np.random.randn(model.cfg.d_model) * 10
        result = rank_one_mlp_edit(model, 0, key_vec, val_vec, scale=5.0)
        orig = np.array(model(tokens))
        edited = np.array(result["edited_model"](tokens))
        assert not np.allclose(orig, edited)


class TestVerifyEditEffect:
    def test_basic(self, model, tokens, metric_fn):
        d_mlp = model.cfg.d_model * 4
        key_vec = np.random.randn(d_mlp)
        val_vec = np.random.randn(model.cfg.d_model)
        edit_result = rank_one_mlp_edit(model, 0, key_vec, val_vec)
        result = verify_edit_effect(model, edit_result["edited_model"], [tokens], metric_fn)
        assert "original_metrics" in result
        assert "edited_metrics" in result
        assert "metric_changes" in result
        assert "mean_change" in result
        assert "success_rate" in result

    def test_lengths(self, model, tokens, metric_fn):
        tokens_list = [tokens, jnp.array([1, 2, 3, 4, 5])]
        result = verify_edit_effect(model, model, tokens_list, metric_fn)
        assert len(result["original_metrics"]) == 2
        assert len(result["metric_changes"]) == 2


class TestEditSideEffects:
    def test_basic(self, model, tokens, metric_fn):
        metric_fns = {
            "m1": metric_fn,
            "m2": lambda logits: float(logits[-1, 2]),
        }
        result = edit_side_effects(model, model, [tokens], metric_fns)
        assert "metric_drifts" in result
        assert "max_drift" in result
        assert "affected_metrics" in result

    def test_same_model_no_drift(self, model, tokens, metric_fn):
        result = edit_side_effects(model, model, [tokens], {"m": metric_fn})
        assert result["max_drift"] < 1e-6


class TestMlpFactExtraction:
    def test_basic(self, model, tokens):
        result = mlp_fact_extraction(model, tokens, layer=0)
        assert "mlp_output_norm" in result
        assert "promoted_tokens" in result
        assert "suppressed_tokens" in result
        assert "output_entropy" in result

    def test_norm_nonneg(self, model, tokens):
        result = mlp_fact_extraction(model, tokens, layer=0)
        assert result["mlp_output_norm"] >= 0
        assert result["output_entropy"] >= 0

    def test_token_count(self, model, tokens):
        result = mlp_fact_extraction(model, tokens, layer=0, top_k=5)
        assert len(result["promoted_tokens"]) == 5
        assert len(result["suppressed_tokens"]) == 5
