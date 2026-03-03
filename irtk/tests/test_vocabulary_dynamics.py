"""Tests for vocabulary dynamics analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.vocabulary_dynamics import (
    embedding_unembed_alignment,
    vocab_subspace_analysis,
    token_frequency_bias,
    embedding_isotropy,
    token_neighborhood_structure,
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


class TestEmbeddingUnembedAlignment:
    def test_returns_dict(self):
        model = _make_model()
        result = embedding_unembed_alignment(model, top_k=5)
        assert "cosine_similarities" in result
        assert "mean_alignment" in result
        assert "top_aligned_tokens" in result
        assert "bottom_aligned_tokens" in result
        assert "alignment_std" in result

    def test_cosines_length(self):
        model = _make_model()
        result = embedding_unembed_alignment(model, top_k=5)
        assert len(result["cosine_similarities"]) == 50

    def test_cosines_bounded(self):
        model = _make_model()
        result = embedding_unembed_alignment(model, top_k=5)
        assert np.all(result["cosine_similarities"] >= -1.0 - 1e-5)
        assert np.all(result["cosine_similarities"] <= 1.0 + 1e-5)

    def test_top_k_count(self):
        model = _make_model()
        result = embedding_unembed_alignment(model, top_k=3)
        assert len(result["top_aligned_tokens"]) == 3
        assert len(result["bottom_aligned_tokens"]) == 3


class TestVocabSubspaceAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        result = vocab_subspace_analysis(model, n_components=3)
        assert "singular_values" in result
        assert "explained_variance_ratio" in result
        assert "cumulative_variance" in result
        assert "effective_rank" in result
        assert "mean_embedding_norm" in result

    def test_sv_count(self):
        model = _make_model()
        result = vocab_subspace_analysis(model, n_components=3)
        assert len(result["singular_values"]) == 3

    def test_variance_sums_to_one(self):
        model = _make_model()
        # With enough components, cumulative should approach 1
        result = vocab_subspace_analysis(model, n_components=16)
        assert result["cumulative_variance"][-1] <= 1.0 + 1e-5

    def test_effective_rank_positive(self):
        model = _make_model()
        result = vocab_subspace_analysis(model, n_components=3)
        assert result["effective_rank"] > 0


class TestTokenFrequencyBias:
    def test_returns_dict(self):
        model = _make_model()
        result = token_frequency_bias(model, top_k=5)
        assert "unembed_norms" in result
        assert "mean_norm" in result
        assert "norm_std" in result
        assert "highest_norm_tokens" in result
        assert "lowest_norm_tokens" in result
        assert "norm_ratio" in result

    def test_norms_length(self):
        model = _make_model()
        result = token_frequency_bias(model, top_k=5)
        assert len(result["unembed_norms"]) == 50

    def test_norms_positive(self):
        model = _make_model()
        result = token_frequency_bias(model, top_k=5)
        assert np.all(result["unembed_norms"] >= 0)

    def test_ratio_ge_one(self):
        model = _make_model()
        result = token_frequency_bias(model, top_k=5)
        assert result["norm_ratio"] >= 1.0 - 1e-5


class TestEmbeddingIsotropy:
    def test_returns_dict(self):
        model = _make_model()
        result = embedding_isotropy(model)
        assert "mean_cosine" in result
        assert "std_cosine" in result
        assert "min_cosine" in result
        assert "max_cosine" in result
        assert "isotropy_score" in result

    def test_isotropy_bounded(self):
        model = _make_model()
        result = embedding_isotropy(model)
        assert 0.0 <= result["isotropy_score"] <= 1.0 + 1e-5

    def test_cosines_bounded(self):
        model = _make_model()
        result = embedding_isotropy(model)
        assert result["min_cosine"] >= -1.0 - 1e-5
        assert result["max_cosine"] <= 1.0 + 1e-5


class TestTokenNeighborhoodStructure:
    def test_returns_dict(self):
        model = _make_model()
        result = token_neighborhood_structure(model, query_tokens=[0, 1, 2], k=3)
        assert "neighbors" in result
        assert "neighbor_similarities" in result
        assert "mean_neighbor_similarity" in result
        assert "self_similarity_rank" in result

    def test_neighbor_count(self):
        model = _make_model()
        result = token_neighborhood_structure(model, query_tokens=[0, 5], k=3)
        assert len(result["neighbors"][0]) == 3
        assert len(result["neighbors"][5]) == 3

    def test_self_excluded(self):
        model = _make_model()
        result = token_neighborhood_structure(model, query_tokens=[0], k=5)
        assert 0 not in result["neighbors"][0]
