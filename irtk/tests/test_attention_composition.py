"""Tests for attention_composition module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_composition import (
    qk_composition_scores,
    v_composition_scores,
    composition_path_tracing,
    virtual_attention_patterns,
    full_composition_matrix,
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


class TestQKCompositionScores:
    def test_basic(self, model):
        result = qk_composition_scores(model)
        assert "q_composition" in result
        assert "k_composition" in result
        assert "top_q_pairs" in result
        assert "top_k_pairs" in result

    def test_shapes(self, model):
        result = qk_composition_scores(model)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["q_composition"].shape == (nl, nh, nl, nh)
        assert result["k_composition"].shape == (nl, nh, nl, nh)

    def test_nonneg(self, model):
        result = qk_composition_scores(model)
        assert np.all(result["q_composition"] >= 0)
        assert np.all(result["k_composition"] >= 0)


class TestVCompositionScores:
    def test_basic(self, model):
        result = v_composition_scores(model)
        assert "v_composition" in result
        assert "top_v_pairs" in result
        assert "mean_v_composition" in result

    def test_shapes(self, model):
        result = v_composition_scores(model)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["v_composition"].shape == (nl, nh, nl, nh)

    def test_nonneg(self, model):
        result = v_composition_scores(model)
        assert np.all(result["v_composition"] >= 0)


class TestCompositionPathTracing:
    def test_basic(self, model, tokens, metric_fn):
        result = composition_path_tracing(model, tokens, metric_fn, max_depth=2)
        assert "path_scores" in result
        assert "top_paths" in result
        assert "n_significant_paths" in result

    def test_has_individual_paths(self, model, tokens, metric_fn):
        result = composition_path_tracing(model, tokens, metric_fn, max_depth=1)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert len(result["path_scores"]) == nl * nh

    def test_scores_nonneg(self, model, tokens, metric_fn):
        result = composition_path_tracing(model, tokens, metric_fn, max_depth=1)
        for path, score in result["path_scores"].items():
            assert score >= 0


class TestVirtualAttentionPatterns:
    def test_basic(self, model, tokens):
        result = virtual_attention_patterns(model, tokens, 0, 0, 1, 0)
        assert "src_pattern" in result
        assert "dst_pattern" in result
        assert "virtual_pattern" in result
        assert "composition_strength" in result

    def test_shapes(self, model, tokens):
        result = virtual_attention_patterns(model, tokens, 0, 0, 1, 0)
        seq_len = len(tokens)
        assert result["src_pattern"].shape == (seq_len, seq_len)
        assert result["dst_pattern"].shape == (seq_len, seq_len)
        assert result["virtual_pattern"].shape == (seq_len, seq_len)

    def test_strength_nonneg(self, model, tokens):
        result = virtual_attention_patterns(model, tokens, 0, 0, 1, 0)
        assert result["composition_strength"] >= 0


class TestFullCompositionMatrix:
    def test_basic(self, model):
        result = full_composition_matrix(model)
        assert "total_composition" in result
        assert "composition_type" in result
        assert "layer_composition_summary" in result
        assert "most_composing_pair" in result

    def test_shapes(self, model):
        result = full_composition_matrix(model)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["total_composition"].shape == (nl, nh, nl, nh)
        assert result["composition_type"].shape == (nl, nh, nl, nh)
        assert result["layer_composition_summary"].shape == (nl, nl)

    def test_pair_valid(self, model):
        result = full_composition_matrix(model)
        sl, sh, dl, dh = result["most_composing_pair"]
        assert 0 <= sl < model.cfg.n_layers
        assert 0 <= sh < model.cfg.n_heads
        assert 0 <= dl < model.cfg.n_layers
        assert 0 <= dh < model.cfg.n_heads
