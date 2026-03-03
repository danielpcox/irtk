"""Tests for residual dynamics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.residual_dynamics import (
    residual_drift_analysis,
    signal_noise_decomposition,
    residual_projection_tracking,
    attention_vs_mlp_contribution_ratio,
    residual_stream_bottleneck,
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


class TestResidualDriftAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_drift_analysis(model, tokens)
        assert "cosine_drift" in result
        assert "cumulative_drift" in result
        assert "max_drift_layer" in result
        assert "total_drift" in result
        assert "norm_trajectory" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_drift_analysis(model, tokens)
        assert len(result["cosine_drift"]) == 2
        assert len(result["norm_trajectory"]) == 3

    def test_drift_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_drift_analysis(model, tokens)
        # Cosine drift = 1 - cos_sim, could be slightly negative due to float precision
        assert np.all(result["cosine_drift"] >= -1e-5)


class TestSignalNoiseDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = signal_noise_decomposition(model, tokens)
        assert "signal_norms" in result
        assert "noise_norms" in result
        assert "signal_fraction" in result
        assert "snr_trajectory" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = signal_noise_decomposition(model, tokens)
        assert len(result["signal_norms"]) == 3
        assert len(result["noise_norms"]) == 3

    def test_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = signal_noise_decomposition(model, tokens)
        assert np.all(result["signal_norms"] >= 0)
        assert np.all(result["noise_norms"] >= -1e-5)


class TestResidualProjectionTracking:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rng = np.random.RandomState(42)
        dirs = rng.randn(3, 16).astype(np.float32)
        result = residual_projection_tracking(model, tokens, dirs)
        assert "projections" in result
        assert "max_projection_layer" in result
        assert "emergence_layer" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        rng = np.random.RandomState(42)
        dirs = rng.randn(3, 16).astype(np.float32)
        result = residual_projection_tracking(model, tokens, dirs)
        assert result["projections"].shape == (3, 3)  # n_dirs x (n_layers+1)
        assert len(result["max_projection_layer"]) == 3


class TestAttentionVsMlpContributionRatio:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_vs_mlp_contribution_ratio(model, tokens)
        assert "attn_norms" in result
        assert "mlp_norms" in result
        assert "attn_ratio" in result
        assert "attn_dominant_layers" in result
        assert "mlp_dominant_layers" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_vs_mlp_contribution_ratio(model, tokens)
        assert len(result["attn_norms"]) == 2
        assert len(result["mlp_norms"]) == 2

    def test_ratio_bounded(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attention_vs_mlp_contribution_ratio(model, tokens)
        assert np.all(result["attn_ratio"] >= 0)
        assert np.all(result["attn_ratio"] <= 1.0 + 1e-5)


class TestResidualStreamBottleneck:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_bottleneck(model, tokens)
        assert "effective_dims" in result
        assert "top_sv_fraction" in result
        assert "bottleneck_layer" in result
        assert "expansion_layer" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_bottleneck(model, tokens)
        assert len(result["effective_dims"]) == 3

    def test_dims_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = residual_stream_bottleneck(model, tokens)
        assert np.all(result["effective_dims"] >= 0)
