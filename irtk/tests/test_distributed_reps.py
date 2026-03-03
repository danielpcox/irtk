"""Tests for distributed representation analysis tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.distributed_reps import (
    linear_representation_probe,
    representation_rank,
    cross_layer_concept_tracking,
    writing_reading_decomposition,
    token_geometry,
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


# ─── Linear Representation Probe ───────────────────────────────────────────


class TestLinearRepresentationProbe:
    def test_returns_dict(self):
        acts = np.random.randn(50, 16).astype(np.float32)
        labels = np.array([0] * 25 + [1] * 25)
        result = linear_representation_probe(acts, labels)
        assert "accuracy" in result
        assert "directions" in result

    def test_direction_shape(self):
        acts = np.random.randn(50, 16).astype(np.float32)
        labels = np.array([0] * 25 + [1] * 25)
        result = linear_representation_probe(acts, labels, n_directions=2)
        assert result["directions"].shape == (2, 16)

    def test_separable_data_high_acc(self):
        # Perfectly separable data
        key = np.random.RandomState(42)
        acts = np.zeros((100, 16), dtype=np.float32)
        acts[:50, 0] = 10.0  # class 0 has high dim-0
        acts[50:, 0] = -10.0  # class 1 has low dim-0
        acts += key.randn(100, 16).astype(np.float32) * 0.1
        labels = np.array([0] * 50 + [1] * 50)
        result = linear_representation_probe(acts, labels)
        assert result["accuracy"] > 0.9

    def test_class_means_shape(self):
        acts = np.random.randn(60, 16).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = linear_representation_probe(acts, labels)
        assert result["class_means"].shape == (3, 16)

    def test_accuracy_in_range(self):
        acts = np.random.randn(50, 16).astype(np.float32)
        labels = np.array([0] * 25 + [1] * 25)
        result = linear_representation_probe(acts, labels)
        assert 0 <= result["accuracy"] <= 1.0


# ─── Representation Rank ───────────────────────────────────────────────────


class TestRepresentationRank:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                jnp.array([8, 9, 10, 11]), jnp.array([12, 13, 14, 15])]
        labels = np.array([0, 0, 1, 1])
        result = representation_rank(model, seqs, "blocks.0.hook_resid_post", labels, max_rank=3)
        assert "accuracies" in result
        assert "estimated_rank" in result

    def test_accuracies_length(self):
        model = _make_model()
        seqs = [jnp.array([i, i+1, i+2, i+3]) for i in range(10)]
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = representation_rank(model, seqs, "blocks.0.hook_resid_post", labels, max_rank=5)
        assert len(result["accuracies"]) == 5

    def test_estimated_rank_positive(self):
        model = _make_model()
        seqs = [jnp.array([i, i+1, i+2, i+3]) for i in range(10)]
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = representation_rank(model, seqs, "blocks.0.hook_resid_post", labels, max_rank=5)
        assert result["estimated_rank"] >= 1


# ─── Cross-Layer Concept Tracking ──────────────────────────────────────────


class TestCrossLayerConceptTracking:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                jnp.array([8, 9, 10, 11]), jnp.array([12, 13, 14, 15])]
        labels = np.array([0, 0, 1, 1])
        result = cross_layer_concept_tracking(model, seqs, labels)
        assert "layer_accuracies" in result
        assert "direction_similarities" in result

    def test_accuracies_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                jnp.array([8, 9, 10, 11]), jnp.array([12, 13, 14, 15])]
        labels = np.array([0, 0, 1, 1])
        result = cross_layer_concept_tracking(model, seqs, labels)
        assert len(result["layer_accuracies"]) == 3  # embed + 2 layers

    def test_labels_correct(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        labels = np.array([0, 1])
        result = cross_layer_concept_tracking(model, seqs, labels)
        assert result["labels"] == ["embed", "L0", "L1"]

    def test_similarities_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]),
                jnp.array([8, 9, 10, 11]), jnp.array([12, 13, 14, 15])]
        labels = np.array([0, 0, 1, 1])
        result = cross_layer_concept_tracking(model, seqs, labels)
        assert len(result["direction_similarities"]) == 2  # n_layers


# ─── Writing-Reading Decomposition ─────────────────────────────────────────


class TestWritingReadingDecomposition:
    def test_returns_dict(self):
        model = _make_model()
        directions = np.random.randn(2, 16).astype(np.float32)
        result = writing_reading_decomposition(model, layer=0, directions=directions)
        assert "writing_scores" in result
        assert "reading_q_scores" in result

    def test_shapes(self):
        model = _make_model()
        directions = np.random.randn(3, 16).astype(np.float32)
        result = writing_reading_decomposition(model, layer=0, directions=directions)
        assert result["writing_scores"].shape == (4, 3)  # n_heads=4, n_dirs=3
        assert result["reading_q_scores"].shape == (4, 3)
        assert result["reading_k_scores"].shape == (4, 3)

    def test_nonnegative(self):
        model = _make_model()
        directions = np.random.randn(2, 16).astype(np.float32)
        result = writing_reading_decomposition(model, layer=0, directions=directions)
        assert np.all(result["writing_scores"] >= 0)
        assert np.all(result["reading_q_scores"] >= 0)

    def test_single_direction(self):
        model = _make_model()
        direction = np.random.randn(16).astype(np.float32)
        result = writing_reading_decomposition(model, layer=0, directions=direction)
        assert result["writing_scores"].shape == (4, 1)


# ─── Token Geometry ─────────────────────────────────────────────────────────


class TestTokenGeometry:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([1, 2, 3, 0])]
        result = token_geometry(model, [0, 1, 2], "blocks.0.hook_resid_post", seqs)
        assert "distance_matrix" in result
        assert "cosine_matrix" in result

    def test_matrix_shapes(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = token_geometry(model, [0, 1], "blocks.0.hook_resid_post", seqs)
        assert result["distance_matrix"].shape == (2, 2)
        assert result["cosine_matrix"].shape == (2, 2)

    def test_self_distance_zero(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = token_geometry(model, [0, 1], "blocks.0.hook_resid_post", seqs)
        assert abs(result["distance_matrix"][0, 0]) < 1e-6
        assert abs(result["distance_matrix"][1, 1]) < 1e-6

    def test_self_cosine_one(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = token_geometry(model, [0, 1], "blocks.0.hook_resid_post", seqs)
        assert abs(result["cosine_matrix"][0, 0] - 1.0) < 0.01
