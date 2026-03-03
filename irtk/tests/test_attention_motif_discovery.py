"""Tests for attention motif discovery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.attention_motif_discovery import (
    extract_attention_motifs,
    motif_prevalence_analysis,
    motif_input_dependency,
    motif_function_inference,
    motif_diversity_by_layer,
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


class TestExtractAttentionMotifs:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_attention_motifs(model, tokens, n_motifs=3)
        assert "motif_patterns" in result
        assert "head_motif_assignments" in result
        assert "head_motif_scores" in result
        assert "n_motifs_found" in result

    def test_assignments_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_attention_motifs(model, tokens, n_motifs=3)
        assert result["head_motif_assignments"].shape == (2, 4)

    def test_motifs_count(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = extract_attention_motifs(model, tokens, n_motifs=3)
        assert result["n_motifs_found"] == 3


class TestMotifPrevalenceAnalysis:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_prevalence_analysis(model, tokens, n_motifs=3)
        assert "motif_counts" in result
        assert "motif_by_layer" in result
        assert "dominant_motif" in result
        assert "motif_diversity" in result

    def test_counts_sum(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_prevalence_analysis(model, tokens, n_motifs=3)
        assert np.sum(result["motif_counts"]) == 8  # 2 layers * 4 heads

    def test_diversity_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_prevalence_analysis(model, tokens, n_motifs=3)
        assert result["motif_diversity"] >= -1e-8


class TestMotifInputDependency:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_input_dependency(model, tokens, layer=0, head=0)
        assert "position_concentration" in result
        assert "diagonal_strength" in result
        assert "recency_strength" in result
        assert "uniformity" in result

    def test_concentration_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_input_dependency(model, tokens, layer=0, head=0)
        assert len(result["position_concentration"]) == 4


class TestMotifFunctionInference:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_function_inference(model, tokens, layer=0, head=0)
        assert "logit_contribution_norm" in result
        assert "top_promoted_tokens" in result
        assert "top_demoted_tokens" in result
        assert "output_direction_norm" in result

    def test_norm_nonneg(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_function_inference(model, tokens, layer=0, head=0)
        assert result["logit_contribution_norm"] >= 0
        assert result["output_direction_norm"] >= 0


class TestMotifDiversityByLayer:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_diversity_by_layer(model, tokens)
        assert "layer_diversity" in result
        assert "most_diverse_layer" in result
        assert "least_diverse_layer" in result
        assert "head_similarities" in result

    def test_diversity_length(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = motif_diversity_by_layer(model, tokens)
        assert len(result["layer_diversity"]) == 2
