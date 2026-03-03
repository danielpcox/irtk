"""Tests for logit attribution decomposition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.logit_attribution_decomposition import (
    full_logit_decomposition,
    per_position_logit_attribution,
    top_promoted_demoted_tokens,
    logit_difference_decomposition,
    cumulative_logit_build,
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


class TestFullLogitDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = full_logit_decomposition(model, tokens)
        assert "embed_contribution" in result
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "total_logit" in result
        assert "reconstruction_error" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = full_logit_decomposition(model, tokens)
        assert result["attn_contributions"].shape == (2, 4)
        assert result["mlp_contributions"].shape == (2,)

    def test_reconstruction(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = full_logit_decomposition(model, tokens)
        # Reconstruction error should be small
        assert result["reconstruction_error"] < 1.0


class TestPerPositionLogitAttribution:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_position_logit_attribution(model, tokens)
        assert "position_contributions" in result
        assert "head_position_contributions" in result
        assert "most_important_position" in result
        assert "position_fraction" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_position_logit_attribution(model, tokens)
        assert len(result["position_contributions"]) == 4
        assert result["head_position_contributions"].shape == (2, 4, 4)

    def test_fractions_sum_to_one(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = per_position_logit_attribution(model, tokens)
        assert abs(np.sum(result["position_fraction"]) - 1.0) < 1e-5


class TestTopPromotedDemotedTokens:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_promoted_demoted_tokens(model, tokens, top_k=3)
        assert "embed_promoted" in result
        assert "embed_demoted" in result
        assert "attn_promoted" in result
        assert "attn_demoted" in result
        assert "mlp_promoted" in result
        assert "mlp_demoted" in result

    def test_promoted_count(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_promoted_demoted_tokens(model, tokens, top_k=3)
        assert len(result["embed_promoted"]) == 3
        assert len(result["embed_demoted"]) == 3

    def test_head_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = top_promoted_demoted_tokens(model, tokens, top_k=3)
        assert (0, 0) in result["attn_promoted"]
        assert 0 in result["mlp_promoted"]


class TestLogitDifferenceDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_difference_decomposition(model, tokens, token_a=5, token_b=10)
        assert "logit_diff" in result
        assert "embed_diff" in result
        assert "attn_diffs" in result
        assert "mlp_diffs" in result
        assert "largest_contributor" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_difference_decomposition(model, tokens, token_a=5, token_b=10)
        assert result["attn_diffs"].shape == (2, 4)
        assert result["mlp_diffs"].shape == (2,)

    def test_decomposition_sums(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = logit_difference_decomposition(model, tokens, token_a=5, token_b=10)
        reconstructed = (
            result["embed_diff"] +
            np.sum(result["attn_diffs"]) +
            np.sum(result["mlp_diffs"]) +
            result["bias_diff"]
        )
        assert abs(reconstructed - result["logit_diff"]) < 1.0


class TestCumulativeLogitBuild:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cumulative_logit_build(model, tokens)
        assert "component_labels" in result
        assert "cumulative_logits" in result
        assert "component_deltas" in result
        assert "final_logit" in result
        assert "biggest_jump_component" in result

    def test_labels_count(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cumulative_logit_build(model, tokens)
        # embed + n_layers * (n_heads + 1 MLP) = 1 + 2 * (4 + 1) = 11
        assert len(result["component_labels"]) == 11

    def test_cumulative_monotonic_final(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = cumulative_logit_build(model, tokens)
        # Final cumulative should equal sum of deltas
        assert abs(result["final_logit"] - np.sum(result["component_deltas"])) < 1e-5
