"""Tests for context sensitivity analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.context_sensitivity import (
    positional_attention_profile,
    local_vs_global_score,
    context_length_sensitivity,
    in_context_learning_dynamics,
    token_distance_effect,
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


# ─── Positional Attention Profile ─────────────────────────────────────────


class TestPositionalAttentionProfile:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = positional_attention_profile(model, seqs)
        assert "profiles" in result
        assert "max_distance" in result
        assert "dominant_distances" in result

    def test_profiles_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = positional_attention_profile(model, seqs)
        assert result["profiles"].shape[0] == 2  # n_layers
        assert result["profiles"].shape[1] == 4  # n_heads

    def test_dominant_distances_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = positional_attention_profile(model, seqs)
        assert result["dominant_distances"].shape == (2, 4)

    def test_profiles_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = positional_attention_profile(model, seqs)
        assert np.all(result["profiles"] >= 0)

    def test_multiple_sequences(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7, 8])]
        result = positional_attention_profile(model, seqs)
        assert result["max_distance"] == 5


# ─── Local vs Global Score ────────────────────────────────────────────────


class TestLocalVsGlobalScore:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = local_vs_global_score(model, seqs, local_window=2)
        assert "local_fractions" in result
        assert "global_fractions" in result
        assert "most_local_heads" in result
        assert "most_global_heads" in result

    def test_fractions_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = local_vs_global_score(model, seqs)
        assert result["local_fractions"].shape == (2, 4)
        assert result["global_fractions"].shape == (2, 4)

    def test_fractions_sum_to_one(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = local_vs_global_score(model, seqs)
        total = result["local_fractions"] + result["global_fractions"]
        np.testing.assert_allclose(total, 1.0, atol=1e-5)

    def test_local_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = local_vs_global_score(model, seqs)
        assert np.all(result["local_fractions"] >= 0)
        assert np.all(result["local_fractions"] <= 1)


# ─── Context Length Sensitivity ───────────────────────────────────────────


class TestContextLengthSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = context_length_sensitivity(model, tokens, _metric)
        assert "lengths" in result
        assert "metrics" in result
        assert "convergence_length" in result
        assert "max_change_length" in result

    def test_lengths_match(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = context_length_sensitivity(model, tokens, _metric)
        assert len(result["lengths"]) == len(result["metrics"])

    def test_lengths_increase(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = context_length_sensitivity(model, tokens, _metric)
        assert np.all(np.diff(result["lengths"]) > 0)

    def test_convergence_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = context_length_sensitivity(model, tokens, _metric)
        assert 1 <= result["convergence_length"] <= 5


# ─── In-Context Learning Dynamics ─────────────────────────────────────────


class TestInContextLearningDynamics:
    def test_returns_dict(self):
        model = _make_model()
        examples = [jnp.array([0, 1]), jnp.array([2, 3])]
        query = jnp.array([4, 5])
        result = in_context_learning_dynamics(model, examples, query, "blocks.0.hook_resid_post")
        assert "n_examples" in result
        assert "representations" in result
        assert "cosine_shifts" in result
        assert "cumulative_shift" in result

    def test_representations_shape(self):
        model = _make_model()
        examples = [jnp.array([0, 1]), jnp.array([2, 3])]
        query = jnp.array([4, 5])
        result = in_context_learning_dynamics(model, examples, query, "blocks.0.hook_resid_post")
        assert result["representations"].shape == (3, 16)  # 3 steps (0, 1, 2 examples), d_model

    def test_cosine_shifts_length(self):
        model = _make_model()
        examples = [jnp.array([0, 1]), jnp.array([2, 3])]
        query = jnp.array([4, 5])
        result = in_context_learning_dynamics(model, examples, query, "blocks.0.hook_resid_post")
        assert len(result["cosine_shifts"]) == 2  # n_examples

    def test_shifts_non_negative(self):
        model = _make_model()
        examples = [jnp.array([0, 1])]
        query = jnp.array([4, 5])
        result = in_context_learning_dynamics(model, examples, query, "blocks.0.hook_resid_post")
        assert np.all(result["cosine_shifts"] >= 0)

    def test_most_impactful_valid(self):
        model = _make_model()
        examples = [jnp.array([0, 1]), jnp.array([2, 3])]
        query = jnp.array([4, 5])
        result = in_context_learning_dynamics(model, examples, query, "blocks.0.hook_resid_post")
        assert 0 <= result["most_impactful_example"] < 2


# ─── Token Distance Effect ───────────────────────────────────────────────


class TestTokenDistanceEffect:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = token_distance_effect(model, tokens, -1, _metric)
        assert "distances" in result
        assert "effects" in result
        assert "effective_window" in result
        assert "peak_distance" in result

    def test_distances_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = token_distance_effect(model, tokens, -1, _metric)
        assert len(result["distances"]) == 5  # all positions up to target

    def test_effects_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = token_distance_effect(model, tokens, -1, _metric)
        assert len(result["effects"]) == len(result["distances"])

    def test_window_non_negative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = token_distance_effect(model, tokens, -1, _metric)
        assert result["effective_window"] >= 0

    def test_peak_distance_valid(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        result = token_distance_effect(model, tokens, -1, _metric)
        assert result["peak_distance"] >= 0
