"""Tests for patching and ablation utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.patching import (
    activation_patch,
    patch_by_layer,
    patch_by_head,
    zero_ablate,
    mean_ablate,
    ablate_heads,
    make_logit_diff_metric,
    make_loss_metric,
    path_patch,
    path_patch_by_receiver,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestActivationPatch:
    def test_basic(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([4, 5, 6, 7])
        metric = lambda logits: float(logits[-1, 0])

        results = activation_patch(
            model, clean, corrupted,
            hook_names=["blocks.0.hook_resid_post"],
            metric_fn=metric,
        )
        assert "blocks.0.hook_resid_post" in results
        assert isinstance(results["blocks.0.hook_resid_post"], float)


class TestPatchByLayer:
    def test_output_shape(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([4, 5, 6, 7])
        metric = lambda logits: float(logits[-1, 0])

        results = patch_by_layer(model, clean, corrupted, metric)
        assert results.shape == (2,)  # n_layers = 2


class TestPatchByHead:
    def test_output_shape(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([4, 5, 6, 7])
        metric = lambda logits: float(logits[-1, 0])

        results = patch_by_head(model, clean, corrupted, metric)
        assert results.shape == (2, 4)  # n_layers x n_heads


class TestZeroAblate:
    def test_basic(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        metric = lambda logits: float(logits[-1, 0])

        results = zero_ablate(
            model, tokens,
            hook_names=["blocks.0.hook_attn_out"],
            metric_fn=metric,
        )
        assert "blocks.0.hook_attn_out" in results


class TestMeanAblate:
    def test_basic(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        metric = lambda logits: float(logits[-1, 0])

        results = mean_ablate(
            model, tokens,
            hook_names=["blocks.0.hook_attn_out"],
            metric_fn=metric,
        )
        assert "blocks.0.hook_attn_out" in results


class TestAblateHeads:
    def test_zero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        metric = lambda logits: float(logits[-1, 0])

        results = ablate_heads(model, tokens, metric, method="zero")
        assert results.shape == (2, 4)

    def test_mean(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        metric = lambda logits: float(logits[-1, 0])

        results = ablate_heads(model, tokens, metric, method="mean")
        assert results.shape == (2, 4)


class TestPathPatch:
    def test_basic(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([4, 5, 6, 7])
        metric = lambda logits: float(logits[-1, 0])

        result = path_patch(
            model, clean, corrupted,
            sender_hook="blocks.0.hook_attn_out",
            receiver_hooks=["blocks.1.attn.hook_q"],
            metric_fn=metric,
        )
        assert isinstance(result, float)

    def test_by_receiver(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([4, 5, 6, 7])
        metric = lambda logits: float(logits[-1, 0])

        results = path_patch_by_receiver(
            model, clean, corrupted,
            sender_hook="blocks.0.hook_attn_out",
            metric_fn=metric,
        )
        # Should have receivers for layer 1's q, k, v
        assert len(results) >= 3
        for name, val in results.items():
            assert isinstance(val, float)


class TestMetrics:
    def test_logit_diff_metric(self):
        metric = make_logit_diff_metric(correct_token=1, wrong_token=2)
        logits = jnp.array([[0.0, 1.0, 0.5]])
        assert metric(logits) == pytest.approx(0.5)

    def test_loss_metric(self):
        metric = make_loss_metric(target_token=0)
        logits = jnp.array([[10.0, 0.0, 0.0]])  # very confident in token 0
        loss = metric(logits)
        assert loss < 0.1  # should be low loss
