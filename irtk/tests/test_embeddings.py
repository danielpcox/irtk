"""Tests for token embedding analysis utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.embeddings import (
    embedding_similarity,
    nearest_neighbors,
    embedding_pca,
    token_analogy,
    embedding_cluster,
    embed_unembed_alignment,
)


def _make_model():
    """Create a model with random weights for embedding analysis."""
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    # Replace zero embeddings with random values
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestEmbeddingSimilarity:
    def test_self_similarity(self):
        model = _make_model()
        sim = embedding_similarity(model, 5, 5)
        assert abs(sim - 1.0) < 1e-5

    def test_range(self):
        model = _make_model()
        sim = embedding_similarity(model, 3, 7)
        assert -1.0 <= sim <= 1.0

    def test_symmetric(self):
        model = _make_model()
        ab = embedding_similarity(model, 3, 7)
        ba = embedding_similarity(model, 7, 3)
        assert abs(ab - ba) < 1e-5

    def test_unembed_space(self):
        model = _make_model()
        sim = embedding_similarity(model, 3, 7, space="unembed")
        assert -1.0 <= sim <= 1.0

    def test_invalid_space(self):
        model = _make_model()
        with pytest.raises(ValueError, match="Unknown space"):
            embedding_similarity(model, 3, 7, space="invalid")


class TestNearestNeighbors:
    def test_self_is_nearest(self):
        model = _make_model()
        results = nearest_neighbors(model, 5, k=5)
        assert results[0][0] == 5
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_returns_k_results(self):
        model = _make_model()
        results = nearest_neighbors(model, 5, k=10)
        assert len(results) == 10

    def test_sorted_descending(self):
        model = _make_model()
        results = nearest_neighbors(model, 5, k=10)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)

    def test_direction_query(self):
        model = _make_model()
        direction = np.array(model.embed.W_E[5])
        results = nearest_neighbors(model, direction, k=5)
        # The direction of token 5's embedding should have token 5 as nearest
        assert results[0][0] == 5

    def test_unembed_space(self):
        model = _make_model()
        results = nearest_neighbors(model, 5, k=5, space="unembed")
        assert len(results) == 5


class TestEmbeddingPCA:
    def test_projections_shape(self):
        model = _make_model()
        result = embedding_pca(model, n_components=3)
        assert result["projections"].shape == (50, 3)

    def test_components_shape(self):
        model = _make_model()
        result = embedding_pca(model, n_components=3)
        assert result["components"].shape == (3, 16)

    def test_explained_variance_sums(self):
        model = _make_model()
        result = embedding_pca(model, n_components=16)
        assert abs(np.sum(result["explained_variance"]) - 1.0) < 1e-5

    def test_subset_tokens(self):
        model = _make_model()
        ids = [0, 5, 10, 15, 20]
        result = embedding_pca(model, token_ids=ids, n_components=2)
        assert result["projections"].shape == (5, 2)
        np.testing.assert_array_equal(result["token_ids"], ids)

    def test_unembed_space(self):
        model = _make_model()
        result = embedding_pca(model, n_components=2, space="unembed")
        assert result["projections"].shape == (50, 2)


class TestTokenAnalogy:
    def test_returns_results(self):
        model = _make_model()
        results = token_analogy(model, a=1, b=2, c=3, k=5)
        assert len(results) == 5

    def test_excludes_input_tokens(self):
        model = _make_model()
        results = token_analogy(model, a=1, b=2, c=3, k=10)
        result_ids = {tid for tid, _ in results}
        assert 1 not in result_ids
        assert 2 not in result_ids
        assert 3 not in result_ids

    def test_similarities_in_range(self):
        model = _make_model()
        results = token_analogy(model, a=1, b=2, c=3, k=5)
        for _, sim in results:
            assert -1.0 <= sim <= 1.0


class TestEmbeddingCluster:
    def test_labels_shape(self):
        model = _make_model()
        ids = list(range(20))
        result = embedding_cluster(model, ids, n_clusters=3)
        assert result["labels"].shape == (20,)

    def test_valid_labels(self):
        model = _make_model()
        ids = list(range(20))
        result = embedding_cluster(model, ids, n_clusters=3)
        assert np.all(result["labels"] >= 0)
        assert np.all(result["labels"] < 3)

    def test_centroids_shape(self):
        model = _make_model()
        ids = list(range(20))
        result = embedding_cluster(model, ids, n_clusters=3)
        assert result["centroids"].shape[0] == 3

    def test_token_ids_returned(self):
        model = _make_model()
        ids = [5, 10, 15, 20, 25]
        result = embedding_cluster(model, ids, n_clusters=2)
        np.testing.assert_array_equal(result["token_ids"], ids)


class TestEmbedUnembedAlignment:
    def test_shape_all_tokens(self):
        model = _make_model()
        result = embed_unembed_alignment(model)
        assert result.shape == (50,)

    def test_shape_subset(self):
        model = _make_model()
        result = embed_unembed_alignment(model, token_ids=[0, 5, 10])
        assert result.shape == (3,)

    def test_range(self):
        model = _make_model()
        result = embed_unembed_alignment(model)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
