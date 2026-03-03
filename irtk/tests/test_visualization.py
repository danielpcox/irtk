"""Tests for visualization module.

We test that all plotting functions run without errors and return
valid matplotlib objects. We use the 'Agg' backend (no display).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp
import pytest

from irtk.visualization import (
    plot_attention_pattern,
    plot_attention_heads,
    plot_head_summary,
    plot_logit_lens,
    plot_residual_norms,
    plot_logit_attribution,
    plot_neuron_activations,
    color_tokens,
    plot_patching_heatmap,
    plot_layer_patching,
    plot_probe_accuracy_by_layer,
    plot_sae_training,
    plot_composition_scores,
    plot_token_attribution,
    plot_causal_tracing,
    plot_prediction_trajectory,
)
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer

import matplotlib.pyplot as plt


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestPlotAttentionPattern:
    def test_basic(self):
        pattern = np.random.rand(5, 5)
        ax = plot_attention_pattern(pattern, title="Test")
        assert ax is not None
        plt.close("all")

    def test_with_jax_array(self):
        pattern = jnp.ones((5, 5)) * 0.2
        ax = plot_attention_pattern(pattern)
        assert ax is not None
        plt.close("all")


class TestPlotAttentionHeads:
    def test_basic(self):
        patterns = np.random.rand(4, 5, 5)
        fig, axes = plot_attention_heads(patterns, layer=0)
        assert fig is not None
        plt.close("all")


class TestPlotHeadSummary:
    def test_basic(self):
        scores = np.random.rand(4, 6)
        fig, ax = plot_head_summary(scores, title="Test Scores")
        assert fig is not None
        plt.close("all")

    def test_no_annotate(self):
        scores = np.random.rand(4, 6)
        fig, ax = plot_head_summary(scores, annotate=False)
        assert fig is not None
        plt.close("all")


class TestPlotLogitLens:
    def test_basic(self):
        per_layer_logits = np.random.randn(3, 5, 50)
        tokens = [0, 1, 2, 3, 4]

        class FakeTokenizer:
            def decode(self, ids):
                return str(ids[0])

        fig, ax = plot_logit_lens(per_layer_logits, tokens, FakeTokenizer())
        assert fig is not None
        plt.close("all")


class TestPlotResidualNorms:
    def test_basic(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        _, cache = model.run_with_cache(tokens)
        fig, ax = plot_residual_norms(cache)
        assert fig is not None
        plt.close("all")


class TestPlotLogitAttribution:
    def test_basic(self):
        attrs = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
        labels = ["embed", "L0 attn", "L0 mlp", "L1 attn", "L1 mlp"]
        fig, ax = plot_logit_attribution(attrs, labels, target_token="hello")
        assert fig is not None
        plt.close("all")


class TestPlotNeuronActivations:
    def test_basic(self):
        acts = np.random.randn(5, 100)
        fig, ax = plot_neuron_activations(acts, n_neurons=30)
        assert fig is not None
        plt.close("all")


class TestColorTokens:
    def test_basic(self):
        tokens = [0, 1, 2, 3]
        values = np.array([0.1, 0.5, 0.9, 0.3])

        class FakeTokenizer:
            def decode(self, ids):
                return f"tok{ids[0]}"

        fig, ax = color_tokens(tokens, values, FakeTokenizer())
        assert fig is not None
        plt.close("all")


class TestPlotPatchingHeatmap:
    def test_basic(self):
        results = np.random.randn(4, 6)
        fig, ax = plot_patching_heatmap(results)
        assert fig is not None
        plt.close("all")

    def test_with_baselines(self):
        results = np.random.randn(4, 6)
        fig, ax = plot_patching_heatmap(
            results, clean_value=1.0, corrupted_value=-0.5
        )
        assert fig is not None
        plt.close("all")


class TestPlotLayerPatching:
    def test_basic(self):
        results = np.random.randn(12)
        fig, ax = plot_layer_patching(results, clean_value=1.0, corrupted_value=0.0)
        assert fig is not None
        plt.close("all")


class TestPlotProbeAccuracy:
    def test_basic(self):
        accs = [0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.88, 0.85]
        fig, ax = plot_probe_accuracy_by_layer(accs)
        assert fig is not None
        plt.close("all")


class TestPlotSAETraining:
    def test_basic(self):
        n = 20
        fig, axes = plot_sae_training(
            train_losses=np.random.rand(n).tolist(),
            recon_losses=np.random.rand(n).tolist(),
            l1_losses=np.random.rand(n).tolist(),
            l0_sparsities=np.random.rand(n).tolist(),
        )
        assert fig is not None
        plt.close("all")


class TestPlotCompositionScores:
    def test_basic(self):
        scores = np.random.rand(8, 8)
        fig, ax = plot_composition_scores(scores, n_heads=4)
        assert fig is not None
        plt.close("all")


class TestPlotTokenAttribution:
    def test_basic(self):
        tokens = [0, 1, 2, 3]
        scores = np.array([0.1, 0.5, 0.9, 0.3])

        class FakeTokenizer:
            def decode(self, ids):
                return f"tok{ids[0]}"

        fig, ax = plot_token_attribution(tokens, scores, FakeTokenizer())
        assert fig is not None
        plt.close("all")

    def test_without_tokenizer(self):
        tokens = ["hello", "world", "!"]
        scores = np.array([0.2, 0.8, 0.1])
        fig, ax = plot_token_attribution(tokens, scores)
        assert fig is not None
        plt.close("all")

    def test_no_values(self):
        tokens = [0, 1, 2]
        scores = np.array([0.5, 0.3, 0.1])
        fig, ax = plot_token_attribution(tokens, scores, show_values=False)
        assert fig is not None
        plt.close("all")

    def test_all_zero_scores(self):
        tokens = [0, 1, 2]
        scores = np.zeros(3)
        fig, ax = plot_token_attribution(tokens, scores)
        assert fig is not None
        plt.close("all")


class TestPlotCausalTracing:
    def test_basic(self):
        result = {
            "clean": 5.0,
            "corrupted": 1.0,
            "restored_resid": np.array([1.5, 2.0, 3.5, 4.8]),
            "restored_attn": np.array([1.2, 1.8, 2.5, 3.0]),
            "restored_mlp": np.array([1.1, 1.5, 2.0, 2.5]),
        }
        fig, ax = plot_causal_tracing(result)
        assert fig is not None
        plt.close("all")


class TestPlotPredictionTrajectory:
    def test_basic(self):
        trajectory = [
            [(5, 0.3), (10, 0.2), (3, 0.1)],
            [(5, 0.5), (10, 0.15), (3, 0.05)],
            [(5, 0.8), (10, 0.1), (7, 0.05)],
        ]
        fig, ax = plot_prediction_trajectory(trajectory)
        assert fig is not None
        plt.close("all")

    def test_with_target_token(self):
        trajectory = [
            [(5, 0.3), (10, 0.2)],
            [(5, 0.8), (10, 0.1)],
        ]
        fig, ax = plot_prediction_trajectory(trajectory, target_token=5)
        assert fig is not None
        plt.close("all")

    def test_empty_trajectory(self):
        fig, ax = plot_prediction_trajectory([])
        assert fig is not None
        plt.close("all")
