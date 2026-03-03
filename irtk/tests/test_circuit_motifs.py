"""Tests for circuit_motifs module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.circuit_motifs import (
    skip_trigram_detection,
    negative_mover_detection,
    backup_circuit_detection,
    signal_boosting_detection,
    motif_catalog,
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


@pytest.fixture
def metric_fn():
    return lambda logits: float(logits[-1, 0] - logits[-1, 1])


class TestSkipTrigramDetection:
    def test_basic(self, model, tokens, metric_fn):
        result = skip_trigram_detection(model, tokens, metric_fn)
        assert "skip_scores" in result
        assert "long_range_heads" in result
        assert "direct_vs_skip" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = skip_trigram_detection(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["skip_scores"].shape == (nl, nh)
        assert result["direct_vs_skip"].shape == (nl, nh)

    def test_scores_nonneg(self, model, tokens, metric_fn):
        result = skip_trigram_detection(model, tokens, metric_fn)
        assert np.all(result["skip_scores"] >= 0)


class TestNegativeMoverDetection:
    def test_basic(self, model, tokens):
        result = negative_mover_detection(model, tokens)
        assert "head_logit_effects" in result
        assert "negative_heads" in result
        assert "positive_heads" in result
        assert "suppression_per_token" in result

    def test_all_heads(self, model, tokens):
        result = negative_mover_detection(model, tokens)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert len(result["head_logit_effects"]) == nl * nh

    def test_suppression_entries(self, model, tokens):
        result = negative_mover_detection(model, tokens)
        for (l, h), tokens_list in result["suppression_per_token"].items():
            assert len(tokens_list) > 0


class TestBackupCircuitDetection:
    def test_basic(self, model, tokens, metric_fn):
        result = backup_circuit_detection(model, tokens, metric_fn)
        assert "single_ablation_effects" in result
        assert "compensation_matrix" in result
        assert "backup_pairs" in result
        assert "redundancy_score" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = backup_circuit_detection(model, tokens, metric_fn)
        nl, nh = model.cfg.n_layers, model.cfg.n_heads
        assert result["single_ablation_effects"].shape == (nl, nh)

    def test_redundancy_range(self, model, tokens, metric_fn):
        result = backup_circuit_detection(model, tokens, metric_fn)
        assert 0 <= result["redundancy_score"] <= 1


class TestSignalBoostingDetection:
    def test_basic(self, model, tokens, metric_fn):
        result = signal_boosting_detection(model, tokens, metric_fn)
        assert "layer_contributions" in result
        assert "cumulative_signal" in result
        assert "signal_trajectory" in result

    def test_shapes(self, model, tokens, metric_fn):
        result = signal_boosting_detection(model, tokens, metric_fn)
        nl = model.cfg.n_layers
        assert result["layer_contributions"].shape == (nl,)
        assert result["signal_trajectory"].shape == (nl,)


class TestMotifCatalog:
    def test_basic(self, model, tokens, metric_fn):
        result = motif_catalog(model, tokens, metric_fn)
        assert "motifs_found" in result
        assert "total_motifs" in result
        assert "dominant_motif" in result
        assert "component_participation" in result

    def test_motif_structure(self, model, tokens, metric_fn):
        result = motif_catalog(model, tokens, metric_fn)
        for motif in result["motifs_found"]:
            assert "type" in motif
            assert "components" in motif
            assert "strength" in motif

    def test_count_consistent(self, model, tokens, metric_fn):
        result = motif_catalog(model, tokens, metric_fn)
        assert result["total_motifs"] == len(result["motifs_found"])
