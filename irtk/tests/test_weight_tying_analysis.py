"""Tests for weight_tying_analysis module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.weight_tying_analysis import (
    embedding_unembed_alignment,
    embedding_subspace_analysis,
    norm_distribution,
    embedding_isotropy,
    token_neighborhood_analysis,
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


class TestEmbeddingUnembedAlignment:
    def test_basic(self, model):
        result = embedding_unembed_alignment(model)
        assert "mean_cosine_similarity" in result
        assert "most_aligned" in result
        assert "is_approximately_tied" in result

    def test_alignment_range(self, model):
        result = embedding_unembed_alignment(model)
        assert -1.5 <= result["mean_cosine_similarity"] <= 1.5


class TestEmbeddingSubspaceAnalysis:
    def test_basic(self, model):
        result = embedding_subspace_analysis(model)
        assert "embedding_effective_rank" in result
        assert "unembed_effective_rank" in result
        assert "subspace_overlap" in result
        assert "shared_dimensions" in result

    def test_ranks_positive(self, model):
        result = embedding_subspace_analysis(model)
        assert result["embedding_effective_rank"] > 0
        assert result["unembed_effective_rank"] > 0


class TestNormDistribution:
    def test_basic(self, model):
        result = norm_distribution(model)
        assert "embedding_norms" in result
        assert "unembed_norms" in result
        assert "norm_correlation" in result

    def test_shapes(self, model):
        result = norm_distribution(model)
        assert result["embedding_norms"].shape == (model.cfg.d_vocab,)


class TestEmbeddingIsotropy:
    def test_basic(self, model):
        result = embedding_isotropy(model)
        assert "isotropy_score" in result
        assert "mean_cosine" in result
        assert "principal_direction_dominance" in result

    def test_isotropy_range(self, model):
        result = embedding_isotropy(model)
        assert 0 <= result["isotropy_score"] <= 2.0


class TestTokenNeighborhoodAnalysis:
    def test_basic(self, model):
        result = token_neighborhood_analysis(model, token_ids=[0, 1, 2])
        assert "per_token" in result
        assert "mean_consistency" in result

    def test_per_token_count(self, model):
        result = token_neighborhood_analysis(model, token_ids=[0, 1, 2])
        assert len(result["per_token"]) == 3
