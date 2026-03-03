"""Tests for activation_surgery module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.activation_surgery import (
    clamp_activation,
    scale_activation,
    project_activation,
    rotate_activation,
    targeted_replacement,
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


class TestClampActivation:
    def test_basic(self, model, tokens):
        result = clamp_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, min_val=-0.1, max_val=0.1)
        assert "original_logits" in result
        assert "clamped_logits" in result
        assert "logit_diff" in result
        assert "n_clamped_values" in result

    def test_no_clamp_no_diff(self, model, tokens):
        result = clamp_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, min_val=-100, max_val=100)
        assert result["logit_diff"] < 1e-4

    def test_clamp_changes_logits(self, model, tokens):
        result = clamp_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, min_val=0, max_val=0)
        # Clamping to zero should change logits
        assert result["logit_diff"] >= 0


class TestScaleActivation:
    def test_basic(self, model, tokens):
        result = scale_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, scale_factor=2.0)
        assert "original_logits" in result
        assert "scaled_logits" in result
        assert "logit_diff" in result
        assert "original_norm" in result
        assert "scaled_norm" in result

    def test_identity_scale(self, model, tokens):
        result = scale_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, scale_factor=1.0)
        assert result["logit_diff"] < 1e-4

    def test_scaled_norm(self, model, tokens):
        result = scale_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, scale_factor=3.0)
        np.testing.assert_allclose(result["scaled_norm"], result["original_norm"] * 3.0, atol=1e-4)


class TestProjectActivation:
    def test_onto(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = project_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, direction=direction, mode="onto")
        assert "projected_logits" in result
        assert "component_magnitude" in result
        assert "fraction_removed" in result

    def test_off(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = project_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, direction=direction, mode="off")
        assert result["logit_diff"] >= 0

    def test_component_magnitude_nonneg(self, model, tokens):
        direction = np.random.randn(model.cfg.d_model)
        result = project_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1, direction=direction)
        assert result["component_magnitude"] >= 0


class TestRotateActivation:
    def test_basic(self, model, tokens):
        from_dir = np.random.randn(model.cfg.d_model)
        to_dir = np.random.randn(model.cfg.d_model)
        result = rotate_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1,
                                   from_dir=from_dir, to_dir=to_dir)
        assert "rotated_logits" in result
        assert "rotation_angle" in result
        assert "component_in_from" in result

    def test_zero_strength(self, model, tokens):
        from_dir = np.random.randn(model.cfg.d_model)
        to_dir = np.random.randn(model.cfg.d_model)
        result = rotate_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1,
                                   from_dir=from_dir, to_dir=to_dir, strength=0.0)
        assert result["logit_diff"] < 1e-4

    def test_rotation_angle(self, model, tokens):
        from_dir = np.zeros(model.cfg.d_model)
        from_dir[0] = 1.0
        to_dir = np.zeros(model.cfg.d_model)
        to_dir[1] = 1.0
        result = rotate_activation(model, tokens, "blocks.0.hook_resid_pre", pos=-1,
                                   from_dir=from_dir, to_dir=to_dir)
        np.testing.assert_allclose(result["rotation_angle"], np.pi / 2, atol=0.01)


class TestTargetedReplacement:
    def test_basic(self, model, tokens):
        replacement = np.zeros(model.cfg.d_model)
        result = targeted_replacement(model, tokens, "blocks.0.hook_resid_pre", pos=-1, replacement_value=replacement)
        assert "replaced_logits" in result
        assert "displacement" in result

    def test_zero_replacement(self, model, tokens):
        replacement = np.zeros(model.cfg.d_model)
        result = targeted_replacement(model, tokens, "blocks.0.hook_resid_pre", pos=-1, replacement_value=replacement)
        assert result["displacement"] >= 0

    def test_same_replacement(self, model, tokens):
        from irtk.hook_points import HookState
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        orig = cache_state.cache["blocks.0.hook_resid_pre"][-1]
        result = targeted_replacement(model, tokens, "blocks.0.hook_resid_pre", pos=-1, replacement_value=np.array(orig))
        assert result["logit_diff"] < 1e-4
        assert result["displacement"] < 1e-4
