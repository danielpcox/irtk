"""Tests for head_circuit_decomposition module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.head_circuit_decomposition import (
    qk_circuit_analysis,
    ov_circuit_analysis,
    virtual_attention_head,
    head_composition_pattern,
    head_logit_contribution,
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


class TestQKCircuitAnalysis:
    def test_basic(self, model):
        result = qk_circuit_analysis(model, layer=0, head=0)
        assert "qk_matrix" in result
        assert "eigenvalues" in result
        assert "effective_rank" in result
        assert "top_query_dirs" in result
        assert "top_key_dirs" in result

    def test_shapes(self, model):
        result = qk_circuit_analysis(model, layer=0, head=0)
        d = model.cfg.d_model
        assert result["qk_matrix"].shape == (d, d)
        assert result["top_query_dirs"].shape[0] == 5
        assert result["top_key_dirs"].shape[0] == 5

    def test_effective_rank_positive(self, model):
        result = qk_circuit_analysis(model, layer=0, head=0)
        assert result["effective_rank"] > 0


class TestOVCircuitAnalysis:
    def test_basic(self, model):
        result = ov_circuit_analysis(model, layer=0, head=0)
        assert "ov_matrix" in result
        assert "singular_values" in result
        assert "effective_rank" in result
        assert "top_input_dirs" in result
        assert "top_output_dirs" in result
        assert "trace" in result

    def test_shapes(self, model):
        result = ov_circuit_analysis(model, layer=0, head=0)
        d = model.cfg.d_model
        assert result["ov_matrix"].shape == (d, d)

    def test_singular_values_ordered(self, model):
        result = ov_circuit_analysis(model, layer=0, head=0)
        svs = result["singular_values"]
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-6


class TestVirtualAttentionHead:
    def test_basic(self, model):
        result = virtual_attention_head(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        assert "qk_composition" in result
        assert "ov_composition" in result
        assert "virtual_ov" in result
        assert "composition_strength" in result

    def test_virtual_ov_shape(self, model):
        result = virtual_attention_head(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        d = model.cfg.d_model
        assert result["virtual_ov"].shape == (d, d)

    def test_composition_nonnegative(self, model):
        result = virtual_attention_head(model, layer_a=0, head_a=0, layer_b=1, head_b=0)
        assert result["qk_composition"] >= 0
        assert result["ov_composition"] >= 0
        assert result["composition_strength"] >= 0


class TestHeadCompositionPattern:
    def test_basic(self, model):
        result = head_composition_pattern(model)
        assert "qk_composition_scores" in result
        assert "ov_composition_scores" in result
        assert "strongest_qk_pair" in result
        assert "strongest_ov_pair" in result
        assert "mean_composition" in result

    def test_shapes(self, model):
        result = head_composition_pattern(model)
        nl = model.cfg.n_layers
        nh = model.cfg.n_heads
        assert result["qk_composition_scores"].shape == (nl, nh, nl, nh)
        assert result["ov_composition_scores"].shape == (nl, nh, nl, nh)

    def test_strongest_pair_format(self, model):
        result = head_composition_pattern(model)
        assert len(result["strongest_qk_pair"]) == 4
        assert len(result["strongest_ov_pair"]) == 4


class TestHeadLogitContribution:
    def test_basic(self, model, tokens):
        result = head_logit_contribution(model, tokens, layer=0, head=0)
        assert "head_output" in result
        assert "logit_contribution" in result
        assert "top_promoted" in result
        assert "top_demoted" in result
        assert "output_norm" in result

    def test_shapes(self, model, tokens):
        result = head_logit_contribution(model, tokens, layer=0, head=0)
        assert result["head_output"].shape == (model.cfg.d_model,)
        assert result["logit_contribution"].shape == (model.cfg.d_vocab,)
        assert len(result["top_promoted"]) == 5
        assert len(result["top_demoted"]) == 5

    def test_output_norm(self, model, tokens):
        result = head_logit_contribution(model, tokens, layer=0, head=0)
        expected_norm = float(np.linalg.norm(result["head_output"]))
        np.testing.assert_allclose(result["output_norm"], expected_norm, atol=1e-4)

    def test_promoted_demoted_order(self, model, tokens):
        result = head_logit_contribution(model, tokens, layer=0, head=0)
        promoted_vals = [v for _, v in result["top_promoted"]]
        demoted_vals = [v for _, v in result["top_demoted"]]
        # Promoted should be descending
        for i in range(len(promoted_vals) - 1):
            assert promoted_vals[i] >= promoted_vals[i + 1] - 1e-6
        # Demoted should be ascending
        for i in range(len(demoted_vals) - 1):
            assert demoted_vals[i] <= demoted_vals[i + 1] + 1e-6
