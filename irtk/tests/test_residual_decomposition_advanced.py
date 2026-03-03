"""Tests for residual_decomposition_advanced module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.residual_decomposition_advanced import (
    orthogonal_decomposition,
    per_token_residual_buildup,
    component_interference,
    residual_subspace_tracking,
    contribution_isolation,
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


class TestOrthogonalDecomposition:
    def test_basic(self, model, tokens):
        result = orthogonal_decomposition(model, tokens)
        assert "components" in result
        assert "explained_variance" in result
        assert "residual_norm" in result

    def test_norm_positive(self, model, tokens):
        result = orthogonal_decomposition(model, tokens)
        assert result["residual_norm"] > 0


class TestPerTokenResidualBuildup:
    def test_basic(self, model, tokens):
        result = per_token_residual_buildup(model, tokens)
        assert "residual_norms" in result
        assert "attn_contributions" in result
        assert "mlp_contributions" in result
        assert "cumulative_buildup" in result

    def test_shapes(self, model, tokens):
        result = per_token_residual_buildup(model, tokens)
        nl = model.cfg.n_layers
        assert result["residual_norms"].shape == (nl,)


class TestComponentInterference:
    def test_basic(self, model, tokens):
        result = component_interference(model, tokens)
        assert "interference_matrix" in result
        assert "constructive_pairs" in result
        assert "destructive_pairs" in result
        assert "net_interference" in result

    def test_labels_populated(self, model, tokens):
        result = component_interference(model, tokens)
        assert len(result["labels"]) > 0


class TestResidualSubspaceTracking:
    def test_basic(self, model, tokens):
        result = residual_subspace_tracking(model, tokens)
        assert "subspace_overlap" in result
        assert "effective_rank" in result

    def test_shapes(self, model, tokens):
        result = residual_subspace_tracking(model, tokens)
        nl = model.cfg.n_layers
        assert result["subspace_overlap"].shape == (nl, nl)
        assert result["effective_rank"].shape == (nl,)


class TestContributionIsolation:
    def test_basic(self, model, tokens):
        result = contribution_isolation(model, tokens, "blocks.0.hook_attn_out")
        assert "contribution_vector" in result
        assert "promoted_tokens" in result
        assert "demoted_tokens" in result

    def test_norm_positive(self, model, tokens):
        result = contribution_isolation(model, tokens, "blocks.0.hook_attn_out")
        assert result["contribution_norm"] > 0
