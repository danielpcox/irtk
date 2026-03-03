"""Tests for activation steering utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.steering import (
    add_vector,
    subtract_vector,
    compute_steering_vector,
    steer_generation,
    activation_diff_at_hook,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


def _make_model_random():
    """Create model with random weights for meaningful results."""
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestAddVector:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vector = jnp.ones(model.cfg.d_model)
        logits = add_vector(model, tokens, "blocks.0.hook_resid_post", vector)
        assert logits.shape == (4, model.cfg.d_vocab)

    def test_changes_output(self):
        model = _make_model_random()
        tokens = jnp.array([0, 1, 2, 3])
        baseline = model(tokens)
        # Use a non-uniform vector (uniform gets killed by LayerNorm mean subtraction)
        key = jax.random.PRNGKey(99)
        vector = jax.random.normal(key, (model.cfg.d_model,)) * 10.0
        steered = add_vector(model, tokens, "hook_embed", vector, alpha=1.0, pos=0)
        assert not np.allclose(baseline, steered, atol=1e-3)

    def test_alpha_zero_is_baseline(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        baseline = model(tokens)
        vector = jnp.ones(model.cfg.d_model)
        steered = add_vector(model, tokens, "blocks.0.hook_resid_post", vector, alpha=0.0)
        np.testing.assert_allclose(baseline, steered, atol=1e-5)

    def test_specific_position(self):
        model = _make_model_random()
        tokens = jnp.array([0, 1, 2, 3])
        key = jax.random.PRNGKey(99)
        vector = jax.random.normal(key, (model.cfg.d_model,)) * 10.0
        # Steering at pos=0 vs pos=2 should give different results
        logits_pos0 = add_vector(model, tokens, "hook_embed", vector, pos=0)
        logits_pos2 = add_vector(model, tokens, "hook_embed", vector, pos=2)
        assert not np.allclose(logits_pos0, logits_pos2, atol=1e-3)


class TestSubtractVector:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vector = jnp.ones(model.cfg.d_model)
        logits = subtract_vector(model, tokens, "hook_embed", vector)
        assert logits.shape == (4, model.cfg.d_vocab)

    def test_changes_output(self):
        model = _make_model_random()
        tokens = jnp.array([0, 1, 2, 3])
        baseline = model(tokens)
        key = jax.random.PRNGKey(99)
        vector = jax.random.normal(key, (model.cfg.d_model,)) * 10.0
        removed = subtract_vector(model, tokens, "hook_embed", vector, pos=0)
        assert not np.allclose(baseline, removed, atol=1e-3)

    def test_specific_position(self):
        model = _make_model_random()
        tokens = jnp.array([0, 1, 2, 3])
        key = jax.random.PRNGKey(99)
        vector = jax.random.normal(key, (model.cfg.d_model,)) * 10.0
        logits_pos0 = subtract_vector(model, tokens, "hook_embed", vector, pos=0)
        logits_all = subtract_vector(model, tokens, "hook_embed", vector)
        assert not np.allclose(logits_pos0, logits_all, atol=1e-5)


class TestComputeSteeringVector:
    def test_output_shape(self):
        model = _make_model()
        pos_tokens = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        neg_tokens = [jnp.array([6, 7, 8]), jnp.array([9, 10, 11])]
        vec = compute_steering_vector(
            model, pos_tokens, neg_tokens, "blocks.0.hook_resid_post"
        )
        assert vec.shape == (model.cfg.d_model,)

    def test_same_inputs_gives_zero(self):
        model = _make_model()
        tokens_list = [jnp.array([0, 1, 2])]
        vec = compute_steering_vector(
            model, tokens_list, tokens_list, "blocks.0.hook_resid_post"
        )
        np.testing.assert_allclose(vec, 0.0, atol=1e-5)

    def test_empty_inputs(self):
        model = _make_model()
        vec = compute_steering_vector(model, [], [], "blocks.0.hook_resid_post")
        assert vec.shape == (model.cfg.d_model,)


class TestSteerGeneration:
    def test_output_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        vector = jnp.zeros(model.cfg.d_model)
        result = steer_generation(
            model, tokens, "blocks.0.hook_resid_post", vector,
            max_new_tokens=5, temperature=0.0,
        )
        assert len(result) == 8  # 3 prompt + 5 generated

    def test_valid_tokens(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        vector = jnp.zeros(model.cfg.d_model)
        result = steer_generation(
            model, tokens, "blocks.0.hook_resid_post", vector,
            max_new_tokens=5, temperature=0.0,
        )
        assert np.all(np.array(result) >= 0)
        assert np.all(np.array(result) < model.cfg.d_vocab)


class TestActivationDiffAtHook:
    def test_output_shape(self):
        model = _make_model()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([4, 5, 6, 7])
        diff = activation_diff_at_hook(
            model, tokens_a, tokens_b, "blocks.0.hook_resid_post"
        )
        assert diff.shape == (model.cfg.d_model,)

    def test_same_input_gives_zero(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        diff = activation_diff_at_hook(
            model, tokens, tokens, "blocks.0.hook_resid_post"
        )
        np.testing.assert_allclose(diff, 0.0, atol=1e-5)

    def test_different_inputs_nonzero(self):
        model = _make_model_random()
        tokens_a = jnp.array([0, 1, 2, 3])
        tokens_b = jnp.array([10, 11, 12, 13])
        diff = activation_diff_at_hook(
            model, tokens_a, tokens_b, "blocks.0.hook_resid_post"
        )
        assert np.linalg.norm(diff) > 1e-5
