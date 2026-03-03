"""Tests for head-level analysis toolkit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.head_analysis import (
    find_previous_token_heads,
    find_induction_heads,
    head_importance_scores,
    classify_heads,
    head_clustering,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


class TestFindPreviousTokenHeads:
    def test_returns_list(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        _, cache = model.run_with_cache(tokens)
        result = find_previous_token_heads(cache, model, threshold=0.0)
        assert isinstance(result, list)

    def test_tuples_have_correct_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        _, cache = model.run_with_cache(tokens)
        result = find_previous_token_heads(cache, model, threshold=0.0)
        for l, h, score in result:
            assert 0 <= l < model.cfg.n_layers
            assert 0 <= h < model.cfg.n_heads
            assert 0.0 <= score <= 1.0

    def test_sorted_descending(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        _, cache = model.run_with_cache(tokens)
        result = find_previous_token_heads(cache, model, threshold=0.0)
        scores = [s for _, _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestFindInductionHeads:
    def test_returns_list(self):
        model = _make_model()
        result = find_induction_heads(model, seq_len=10, threshold=0.0)
        assert isinstance(result, list)

    def test_high_threshold_fewer_results(self):
        model = _make_model()
        low = find_induction_heads(model, seq_len=10, threshold=0.0)
        high = find_induction_heads(model, seq_len=10, threshold=0.9)
        assert len(high) <= len(low)


class TestHeadImportanceScores:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = head_importance_scores(model, tokens)
        assert "entropy" in result
        assert "max_attn" in result
        assert "prev_token" in result
        assert "diag_score" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = head_importance_scores(model, tokens)
        for key in ["entropy", "max_attn", "prev_token", "diag_score"]:
            assert result[key].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_with_target_token(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = head_importance_scores(model, tokens, target_token=5)
        assert "direct_logit" in result
        assert result["direct_logit"].shape == (model.cfg.n_layers, model.cfg.n_heads)

    def test_entropy_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = head_importance_scores(model, tokens)
        assert np.all(result["entropy"] >= 0)


class TestClassifyHeads:
    def test_returns_expected_categories(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = classify_heads(model, tokens)
        expected_keys = {"previous_token", "self_attention", "bos_attention", "broad", "other"}
        assert set(result.keys()) == expected_keys

    def test_all_heads_classified(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = classify_heads(model, tokens)
        total = sum(len(v) for v in result.values())
        assert total == model.cfg.n_layers * model.cfg.n_heads

    def test_no_duplicates(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = classify_heads(model, tokens)
        all_heads = []
        for heads in result.values():
            all_heads.extend(heads)
        assert len(all_heads) == len(set(all_heads))


class TestHeadClustering:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = head_clustering(model, tokens, n_clusters=3)
        assert "labels" in result
        assert "features" in result
        assert "feature_names" in result

    def test_labels_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = head_clustering(model, tokens, n_clusters=3)
        total = model.cfg.n_layers * model.cfg.n_heads
        assert result["labels"].shape == (total,)

    def test_valid_cluster_labels(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        n_clusters = 3
        result = head_clustering(model, tokens, n_clusters=n_clusters)
        assert np.all(result["labels"] >= 0)
        assert np.all(result["labels"] < n_clusters)

    def test_features_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = head_clustering(model, tokens, n_clusters=3)
        total = model.cfg.n_layers * model.cfg.n_heads
        assert result["features"].shape == (total, 4)
