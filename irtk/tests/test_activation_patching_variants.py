"""Tests for activation patching variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.activation_patching_variants import (
    denoising_patching,
    noising_patching,
    mean_ablation,
    resample_ablation,
    causal_mediation_analysis,
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


class TestDenoisingPatching:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = denoising_patching(model, corrupted, clean, _metric)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "baseline_metric" in result
        assert "clean_metric" in result
        assert "recovery_fractions" in result

    def test_shapes(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = denoising_patching(model, corrupted, clean, _metric)
        assert result["attn_effects"].shape == (2, 4)
        assert result["mlp_effects"].shape == (2,)

    def test_recovery_fractions_keys(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = denoising_patching(model, corrupted, clean, _metric)
        # Should have entries for all heads and MLPs
        assert len(result["recovery_fractions"]) == 2 * 4 + 2


class TestNoisingPatching:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = noising_patching(model, clean, corrupted, _metric)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "baseline_metric" in result
        assert "corrupted_metric" in result
        assert "damage_fractions" in result

    def test_shapes(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = noising_patching(model, clean, corrupted, _metric)
        assert result["attn_effects"].shape == (2, 4)
        assert result["mlp_effects"].shape == (2,)


class TestMeanAblation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mean_ablation(model, tokens, _metric, n_samples=2)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "baseline_metric" in result
        assert "most_critical_component" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mean_ablation(model, tokens, _metric, n_samples=2)
        assert result["attn_effects"].shape == (2, 4)
        assert result["mlp_effects"].shape == (2,)

    def test_critical_is_tuple(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = mean_ablation(model, tokens, _metric, n_samples=2)
        assert isinstance(result["most_critical_component"], tuple)


class TestResampleAblation:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = resample_ablation(model, tokens, _metric, n_resamples=2)
        assert "attn_effects" in result
        assert "mlp_effects" in result
        assert "attn_std" in result
        assert "mlp_std" in result
        assert "baseline_metric" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = resample_ablation(model, tokens, _metric, n_resamples=2)
        assert result["attn_effects"].shape == (2, 4)
        assert result["attn_std"].shape == (2, 4)
        assert result["mlp_effects"].shape == (2,)
        assert result["mlp_std"].shape == (2,)

    def test_std_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = resample_ablation(model, tokens, _metric, n_resamples=2)
        assert np.all(result["attn_std"] >= 0)
        assert np.all(result["mlp_std"] >= 0)


class TestCausalMediationAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = causal_mediation_analysis(model, clean, corrupted, _metric, mediator_layer=0)
        assert "total_effect" in result
        assert "indirect_effect" in result
        assert "direct_effect" in result
        assert "mediation_fraction" in result
        assert "attn_indirect" in result
        assert "mlp_indirect" in result

    def test_effect_decomposition(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        result = causal_mediation_analysis(model, clean, corrupted, _metric, mediator_layer=0)
        # direct + indirect should approximately equal total
        assert abs(result["direct_effect"] + result["indirect_effect"] - result["total_effect"]) < 1e-4

    def test_different_layers(self):
        model = _make_model()
        clean = jnp.array([0, 1, 2, 3])
        corrupted = jnp.array([5, 6, 7, 8])
        r0 = causal_mediation_analysis(model, clean, corrupted, _metric, mediator_layer=0)
        r1 = causal_mediation_analysis(model, clean, corrupted, _metric, mediator_layer=1)
        # Total effect should be the same regardless of mediator
        assert abs(r0["total_effect"] - r1["total_effect"]) < 1e-4
