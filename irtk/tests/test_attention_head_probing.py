"""Tests for attention_head_probing module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_probing import (
    head_positional_probe,
    head_token_identity_probe,
    head_specialization_profile,
    head_output_norm_analysis,
    head_value_rank_analysis,
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
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokens():
    return jnp.array([0, 5, 10, 15, 20, 25, 30, 35])


class TestHeadPositionalProbe:
    def test_output_keys(self, model, tokens):
        r = head_positional_probe(model, tokens)
        assert "distance_correlation" in r
        assert "recency_score" in r
        assert "locality_score" in r
        assert "most_positional_head" in r
        assert "least_positional_head" in r

    def test_shapes(self, model, tokens):
        r = head_positional_probe(model, tokens)
        assert r["distance_correlation"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["recency_score"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["locality_score"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_locality_bounded(self, model, tokens):
        r = head_positional_probe(model, tokens)
        assert np.all(r["locality_score"] >= -1e-5)
        assert np.all(r["locality_score"] <= 1.0 + 1e-5)


class TestHeadTokenIdentityProbe:
    def test_output_keys(self, model, tokens):
        r = head_token_identity_probe(model, tokens)
        assert "identity_scores" in r
        assert "transformation_scores" in r
        assert "most_identity_preserving" in r
        assert "most_transforming" in r

    def test_shapes(self, model, tokens):
        r = head_token_identity_probe(model, tokens)
        assert r["identity_scores"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["transformation_scores"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_identity_bounded(self, model, tokens):
        r = head_token_identity_probe(model, tokens)
        assert np.all(r["identity_scores"] >= -1.0 - 1e-5)
        assert np.all(r["identity_scores"] <= 1.0 + 1e-5)


class TestHeadSpecializationProfile:
    def test_output_keys(self, model, tokens):
        r = head_specialization_profile(model, tokens)
        assert "entropy_scores" in r
        assert "bos_scores" in r
        assert "diagonal_scores" in r
        assert "prev_token_scores" in r
        assert "specialization_labels" in r

    def test_shapes(self, model, tokens):
        r = head_specialization_profile(model, tokens)
        assert r["entropy_scores"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["specialization_labels"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_labels_valid(self, model, tokens):
        r = head_specialization_profile(model, tokens)
        valid_labels = {"bos", "prev_token", "identity", "uniform", "mixed", "unknown"}
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                assert r["specialization_labels"][l, h] in valid_labels

    def test_scores_nonneg(self, model, tokens):
        r = head_specialization_profile(model, tokens)
        assert np.all(r["entropy_scores"] >= -1e-5)
        assert np.all(r["bos_scores"] >= -1e-5)


class TestHeadOutputNormAnalysis:
    def test_output_keys(self, model, tokens):
        r = head_output_norm_analysis(model, tokens)
        assert "output_norms" in r
        assert "relative_norms" in r
        assert "largest_head" in r
        assert "smallest_head" in r
        assert "layer_total_norms" in r

    def test_shapes(self, model, tokens):
        r = head_output_norm_analysis(model, tokens)
        assert r["output_norms"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["relative_norms"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["layer_total_norms"].shape == (model.cfg.n_layers,)

    def test_norms_nonneg(self, model, tokens):
        r = head_output_norm_analysis(model, tokens)
        assert np.all(r["output_norms"] >= 0)

    def test_relative_sum(self, model, tokens):
        r = head_output_norm_analysis(model, tokens)
        for l in range(model.cfg.n_layers):
            assert abs(np.sum(r["relative_norms"][l]) - 1.0) < 1e-4


class TestHeadValueRankAnalysis:
    def test_output_keys(self, model, tokens):
        r = head_value_rank_analysis(model, tokens, top_k=3)
        assert "effective_ranks" in r
        assert "top_singular_values" in r
        assert "most_low_rank" in r
        assert "most_full_rank" in r
        assert "mean_rank" in r

    def test_shapes(self, model, tokens):
        r = head_value_rank_analysis(model, tokens, top_k=3)
        assert r["effective_ranks"].shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert r["top_singular_values"].shape == (model.cfg.n_layers, model.cfg.n_heads, 3)

    def test_ranks_positive(self, model, tokens):
        r = head_value_rank_analysis(model, tokens, top_k=3)
        assert np.all(r["effective_ranks"] > 0)
        assert r["mean_rank"] > 0
