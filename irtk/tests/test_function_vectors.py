"""Tests for function vector analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.function_vectors import (
    extract_function_vector,
    scan_for_function_heads,
    function_vector_transfer,
    function_vector_arithmetic,
    function_vector_similarity_matrix,
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


class TestExtractFunctionVector:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_function_vector(model, tokens, layer=0, head=0)
        assert "function_vector" in result
        assert "fv_norm" in result
        assert "head_output_norm" in result
        assert "extraction_position" in result

    def test_fv_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_function_vector(model, tokens, layer=0, head=0)
        assert result["function_vector"].shape == (16,)

    def test_norm_positive(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_function_vector(model, tokens, layer=0, head=0)
        assert result["fv_norm"] > 0


class TestScanForFunctionHeads:
    def test_returns_dict(self):
        model = _make_model()
        few_shot = jnp.array([0, 1, 2, 3])
        zero_shot = jnp.array([4, 5, 6, 7])
        result = scan_for_function_heads(model, few_shot, zero_shot, _metric, top_k=3)
        assert "transfer_scores" in result
        assert "top_function_heads" in result
        assert "baseline_metric" in result

    def test_scores_shape(self):
        model = _make_model()
        few_shot = jnp.array([0, 1, 2, 3])
        zero_shot = jnp.array([4, 5, 6, 7])
        result = scan_for_function_heads(model, few_shot, zero_shot, _metric)
        assert result["transfer_scores"].shape == (2, 4)

    def test_top_k_length(self):
        model = _make_model()
        few_shot = jnp.array([0, 1, 2, 3])
        zero_shot = jnp.array([4, 5, 6, 7])
        result = scan_for_function_heads(model, few_shot, zero_shot, _metric, top_k=3)
        assert len(result["top_function_heads"]) == 3


class TestFunctionVectorTransfer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv = np.random.randn(16).astype(np.float32) * 0.1
        result = function_vector_transfer(model, fv, tokens, inject_layer=0)
        assert "patched_logits" in result
        assert "clean_logits" in result
        assert "logit_diff" in result
        assert "top_promoted" in result

    def test_logit_diff_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv = np.random.randn(16).astype(np.float32) * 0.1
        result = function_vector_transfer(model, fv, tokens, inject_layer=0)
        assert len(result["logit_diff"]) == 50  # d_vocab

    def test_top_promoted_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv = np.random.randn(16).astype(np.float32) * 0.1
        result = function_vector_transfer(model, fv, tokens, inject_layer=0)
        assert len(result["top_promoted"]) == 5


class TestFunctionVectorArithmetic:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv_a = np.random.randn(16).astype(np.float32) * 0.1
        fv_b = np.random.randn(16).astype(np.float32) * 0.1
        result = function_vector_arithmetic(model, fv_a, fv_b, tokens, inject_layer=0)
        assert "combined_logits" in result
        assert "fv_a_logits" in result
        assert "fv_b_logits" in result
        assert "linearity_score" in result
        assert "cosine_similarity" in result

    def test_linearity_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv_a = np.random.randn(16).astype(np.float32) * 0.01
        fv_b = np.random.randn(16).astype(np.float32) * 0.01
        result = function_vector_arithmetic(model, fv_a, fv_b, tokens, inject_layer=0)
        assert result["linearity_score"] >= 0

    def test_cosine_in_range(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        fv_a = np.random.randn(16).astype(np.float32) * 0.1
        fv_b = np.random.randn(16).astype(np.float32) * 0.1
        result = function_vector_arithmetic(model, fv_a, fv_b, tokens, inject_layer=0)
        assert -1.0 <= result["cosine_similarity"] <= 1.0


class TestFunctionVectorSimilarityMatrix:
    def test_returns_dict(self):
        model = _make_model()
        tasks = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7]), jnp.array([8, 9, 10, 11])]
        result = function_vector_similarity_matrix(model, tasks, layer=0, head=0)
        assert "similarity_matrix" in result
        assert "function_vectors" in result
        assert "norms" in result
        assert "mean_similarity" in result

    def test_matrix_shape(self):
        model = _make_model()
        tasks = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = function_vector_similarity_matrix(model, tasks, layer=0, head=0)
        assert result["similarity_matrix"].shape == (2, 2)

    def test_diagonal_ones(self):
        model = _make_model()
        tasks = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = function_vector_similarity_matrix(model, tasks, layer=0, head=0)
        for i in range(2):
            assert np.isclose(result["similarity_matrix"][i, i], 1.0, atol=0.01)
