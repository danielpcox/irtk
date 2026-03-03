"""Tests for layer_communication module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.layer_communication import (
    layer_message_norms,
    channel_utilization,
    bandwidth_analysis,
    layer_bypass_detection,
    inter_layer_alignment,
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


class TestLayerMessageNorms:
    def test_basic(self, model, tokens):
        result = layer_message_norms(model, tokens)
        assert "attn_message_norms" in result
        assert "mlp_message_norms" in result
        assert "residual_norms" in result
        assert "message_to_residual_ratio" in result

    def test_shapes(self, model, tokens):
        result = layer_message_norms(model, tokens)
        nl = model.cfg.n_layers
        assert result["attn_message_norms"].shape == (nl,)
        assert result["mlp_message_norms"].shape == (nl,)
        assert result["residual_norms"].shape == (nl + 1,)

    def test_norms_nonneg(self, model, tokens):
        result = layer_message_norms(model, tokens)
        assert np.all(result["attn_message_norms"] >= 0)
        assert np.all(result["mlp_message_norms"] >= 0)


class TestChannelUtilization:
    def test_basic(self, model, tokens):
        result = channel_utilization(model, tokens)
        assert "effective_dims" in result
        assert "utilization" in result
        assert "top_dim_fraction" in result

    def test_shapes(self, model, tokens):
        result = channel_utilization(model, tokens)
        nl = model.cfg.n_layers
        assert result["effective_dims"].shape == (nl + 1,)
        assert result["utilization"].shape == (nl + 1,)

    def test_utilization_range(self, model, tokens):
        result = channel_utilization(model, tokens)
        assert np.all(result["utilization"] >= 0)
        assert np.all(result["utilization"] <= 1.01)


class TestBandwidthAnalysis:
    def test_basic(self, model, tokens):
        result = bandwidth_analysis(model, tokens)
        assert "layer_deltas" in result
        assert "delta_ranks" in result
        assert "bandwidth" in result
        assert "bottleneck_layer" in result

    def test_shapes(self, model, tokens):
        result = bandwidth_analysis(model, tokens)
        nl = model.cfg.n_layers
        d = model.cfg.d_model
        assert result["layer_deltas"].shape == (nl, d)
        assert result["bandwidth"].shape == (nl,)

    def test_bottleneck_valid(self, model, tokens):
        result = bandwidth_analysis(model, tokens)
        assert 0 <= result["bottleneck_layer"] < model.cfg.n_layers


class TestLayerBypassDetection:
    def test_basic(self, model, tokens, metric_fn):
        result = layer_bypass_detection(model, tokens, metric_fn)
        assert "skip_similarity" in result
        assert "metric_impact" in result
        assert "is_bypass" in result
        assert "n_bypass_layers" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = layer_bypass_detection(model, tokens, metric_fn)
        nl = model.cfg.n_layers
        assert result["skip_similarity"].shape == (nl,)
        assert result["metric_impact"].shape == (nl,)
        assert result["is_bypass"].shape == (nl,)

    def test_n_bypass_valid(self, model, tokens, metric_fn):
        result = layer_bypass_detection(model, tokens, metric_fn)
        assert 0 <= result["n_bypass_layers"] <= model.cfg.n_layers


class TestInterLayerAlignment:
    def test_basic(self, model, tokens):
        result = inter_layer_alignment(model, tokens)
        assert "cosine_alignment" in result
        assert "attn_mlp_alignment" in result
        assert "mean_alignment" in result

    def test_shapes(self, model, tokens):
        result = inter_layer_alignment(model, tokens)
        nl = model.cfg.n_layers
        assert result["cosine_alignment"].shape == (nl,)
        assert result["attn_mlp_alignment"].shape == (nl,)

    def test_alignment_range(self, model, tokens):
        result = inter_layer_alignment(model, tokens)
        assert np.all(result["cosine_alignment"] >= -1.01)
        assert np.all(result["cosine_alignment"] <= 1.01)
