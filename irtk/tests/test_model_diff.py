"""Tests for model comparison and diffing utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.model_diff import (
    weight_diff,
    weight_diff_summary,
    activation_diff,
    logit_diff_on_dataset,
    finetuning_attribution,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


def _perturb_model(model):
    """Create a slightly perturbed copy of the model."""
    key = jax.random.PRNGKey(99)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(leaf + jax.random.normal(subkey, leaf.shape) * 0.01)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestWeightDiff:
    def test_identical_models(self):
        """Diff of a model with itself should be zero."""
        model = _make_model()
        diffs = weight_diff(model, model)
        for name, d in diffs.items():
            np.testing.assert_allclose(d["abs_diff"], 0.0, atol=1e-10)

    def test_perturbed_model(self):
        """Diff with perturbed model should be non-zero."""
        model = _make_model()
        perturbed = _perturb_model(model)
        diffs = weight_diff(model, perturbed)
        some_nonzero = any(d["abs_diff"] > 1e-6 for d in diffs.values())
        assert some_nonzero

    def test_expected_keys(self):
        model = _make_model()
        diffs = weight_diff(model, model)
        assert "embed.W_E" in diffs
        assert "unembed.W_U" in diffs
        assert "blocks.0.attn.W_Q" in diffs
        assert "blocks.0.mlp.W_in" in diffs
        assert "blocks.1.attn.W_O" in diffs

    def test_dict_structure(self):
        model = _make_model()
        diffs = weight_diff(model, model)
        for name, d in diffs.items():
            assert "abs_diff" in d
            assert "rel_diff" in d
            assert "norm_a" in d


class TestWeightDiffSummary:
    def test_returns_top_k(self):
        model = _make_model()
        perturbed = _perturb_model(model)
        summary = weight_diff_summary(model, perturbed, top_k=5)
        assert len(summary) == 5

    def test_sorted_by_rel_diff(self):
        model = _make_model()
        perturbed = _perturb_model(model)
        summary = weight_diff_summary(model, perturbed, top_k=10)
        rel_diffs = [s[2] for s in summary]
        assert rel_diffs == sorted(rel_diffs, reverse=True)


class TestActivationDiff:
    def test_identical_models(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        diffs = activation_diff(model, model, tokens)
        for name, val in diffs.items():
            np.testing.assert_allclose(val, 0.0, atol=1e-6)

    def test_perturbed_model(self):
        model = _make_model()
        perturbed = _perturb_model(model)
        tokens = jnp.array([0, 1, 2, 3])
        diffs = activation_diff(model, perturbed, tokens)
        some_nonzero = any(v > 1e-6 for v in diffs.values())
        assert some_nonzero


class TestLogitDiffOnDataset:
    def test_identical_models(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        result = logit_diff_on_dataset(model, model, seqs)
        np.testing.assert_allclose(result["logit_l2"], 0.0, atol=1e-5)
        np.testing.assert_allclose(result["kl_divergence"], 0.0, atol=1e-5)
        assert all(result["top1_agree"])

    def test_expected_keys(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        result = logit_diff_on_dataset(model, model, seqs)
        assert "logit_l2" in result
        assert "kl_divergence" in result
        assert "top1_agree" in result
        assert "mean_logit_l2" in result

    def test_shapes(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5]), jnp.array([6, 7, 8])]
        result = logit_diff_on_dataset(model, model, seqs)
        assert result["logit_l2"].shape == (3,)
        assert result["kl_divergence"].shape == (3,)
        assert result["top1_agree"].shape == (3,)


class TestFinetuningAttribution:
    def test_identical_models(self):
        model = _make_model()
        result = finetuning_attribution(model, model)
        np.testing.assert_allclose(result["head_diff"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["mlp_diff"], 0.0, atol=1e-10)

    def test_shapes(self):
        model = _make_model()
        perturbed = _perturb_model(model)
        result = finetuning_attribution(model, perturbed)
        assert result["head_diff"].shape == (2, 4)  # n_layers, n_heads
        assert result["mlp_diff"].shape == (2,)
        assert result["ln_diff"].shape == (2,)

    def test_perturbed_nonzero(self):
        model = _make_model()
        perturbed = _perturb_model(model)
        result = finetuning_attribution(model, perturbed)
        assert np.sum(result["head_diff"]) > 0
        assert np.sum(result["mlp_diff"]) > 0
