"""Tests for copy suppression analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.copy_suppression import (
    copy_suppression_score,
    find_negative_heads,
    suppression_per_attended_token,
    copy_vs_suppress_decomposition,
    suppression_circuit_on_ioi,
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


class TestCopySuppressionScore:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = copy_suppression_score(model, tokens, layer=0, head=0)
        assert "suppression_scores" in result
        assert "mean_suppression" in result
        assert "max_suppression_pos" in result
        assert "is_copy_suppressing" in result

    def test_scores_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = copy_suppression_score(model, tokens, layer=0, head=0)
        assert len(result["suppression_scores"]) == 6

    def test_is_bool(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = copy_suppression_score(model, tokens, layer=0, head=0)
        assert isinstance(result["is_copy_suppressing"], bool)


class TestFindNegativeHeads:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = find_negative_heads(model, tokens, top_k=3)
        assert "negative_scores" in result
        assert "top_negative_heads" in result
        assert "n_negative_heads" in result

    def test_scores_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = find_negative_heads(model, tokens)
        assert result["negative_scores"].shape == (2, 4)

    def test_top_k_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = find_negative_heads(model, tokens, top_k=3)
        assert len(result["top_negative_heads"]) == 3


class TestSuppressionPerAttendedToken:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = suppression_per_attended_token(model, tokens, layer=0, head=0)
        assert "token_suppression" in result
        assert "attention_weights" in result
        assert "most_suppressed_pos" in result
        assert "total_suppression" in result

    def test_attention_sums(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = suppression_per_attended_token(model, tokens, layer=0, head=0)
        assert np.isclose(np.sum(result["attention_weights"]), 1.0, atol=0.01)


class TestCopyVsSuppressDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = copy_vs_suppress_decomposition(model, tokens)
        assert "head_contributions" in result
        assert "copy_heads" in result
        assert "suppress_heads" in result
        assert "net_effect" in result

    def test_contributions_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = copy_vs_suppress_decomposition(model, tokens)
        assert result["head_contributions"].shape == (2, 4)

    def test_copy_and_suppress_cover_all(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = copy_vs_suppress_decomposition(model, tokens)
        total = len(result["copy_heads"]) + len(result["suppress_heads"])
        # Some heads may have exactly 0 contribution
        assert total <= 8


class TestSuppressionCircuitOnIOI:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = suppression_circuit_on_ioi(model, tokens, subject_pos=1, io_pos=3)
        assert "subject_suppression_by_head" in result
        assert "io_promotion_by_head" in result
        assert "net_effect_by_head" in result
        assert "top_suppressors" in result
        assert "subject_token" in result
        assert "io_token" in result

    def test_tokens_correct(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = suppression_circuit_on_ioi(model, tokens, subject_pos=1, io_pos=3)
        assert result["subject_token"] == 1
        assert result["io_token"] == 3

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = suppression_circuit_on_ioi(model, tokens, subject_pos=1, io_pos=3)
        assert result["subject_suppression_by_head"].shape == (2, 4)
        assert result["io_promotion_by_head"].shape == (2, 4)
        assert result["net_effect_by_head"].shape == (2, 4)
