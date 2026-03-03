"""Tests for automated interpretability tools."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.auto_interp import (
    auto_label_head,
    auto_label_neuron,
    feature_summary_stats,
    head_type_classifier,
    component_report,
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


# ─── Auto Label Head ─────────────────────────────────────────────────────


class TestAutoLabelHead:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = auto_label_head(model, 0, 0, seqs)
        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert "mean_entropy" in result

    def test_label_is_string(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = auto_label_head(model, 0, 0, seqs)
        assert isinstance(result["label"], str)

    def test_confidence_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = auto_label_head(model, 0, 0, seqs)
        assert 0 <= result["confidence"] <= 1.0

    def test_scores_has_types(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = auto_label_head(model, 0, 0, seqs)
        assert "previous_token" in result["scores"]
        assert "current_token" in result["scores"]

    def test_multiple_sequences(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
        result = auto_label_head(model, 1, 2, seqs)
        assert result["label"] != ""

    def test_entropy_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5])]
        result = auto_label_head(model, 0, 0, seqs)
        assert result["mean_entropy"] >= 0


# ─── Auto Label Neuron ───────────────────────────────────────────────────


class TestAutoLabelNeuron:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = auto_label_neuron(model, 0, 0, seqs)
        assert "top_activations" in result
        assert "mean_activation" in result
        assert "firing_rate" in result
        assert "max_activation" in result

    def test_top_activations_limited(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
        result = auto_label_neuron(model, 0, 0, seqs, k=5)
        assert len(result["top_activations"]) <= 5

    def test_firing_rate_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = auto_label_neuron(model, 0, 0, seqs)
        assert 0 <= result["firing_rate"] <= 1.0

    def test_max_gte_mean(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = auto_label_neuron(model, 0, 0, seqs)
        assert result["max_activation"] >= result["mean_activation"]


# ─── Feature Summary Stats ───────────────────────────────────────────────


class TestFeatureSummaryStats:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_summary_stats(model, "blocks.0.hook_resid_post", seqs)
        assert "mean_activations" in result
        assert "std_activations" in result
        assert "sparsity" in result
        assert "kurtosis" in result

    def test_shapes(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_summary_stats(model, "blocks.0.hook_resid_post", seqs)
        d = model.cfg.d_model
        assert len(result["mean_activations"]) == d
        assert len(result["std_activations"]) == d

    def test_std_non_negative(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_summary_stats(model, "blocks.0.hook_resid_post", seqs)
        assert np.all(result["std_activations"] >= 0)

    def test_sparsity_in_range(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = feature_summary_stats(model, "blocks.0.hook_resid_post", seqs)
        assert np.all(result["sparsity"] >= 0)
        assert np.all(result["sparsity"] <= 1)


# ─── Head Type Classifier ────────────────────────────────────────────────


class TestHeadTypeClassifier:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_type_classifier(model, seqs)
        assert "classifications" in result
        assert "type_counts" in result
        assert "confidence_matrix" in result

    def test_classifications_count(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_type_classifier(model, seqs)
        assert len(result["classifications"]) == 8  # 2 layers * 4 heads

    def test_confidence_matrix_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_type_classifier(model, seqs)
        assert result["confidence_matrix"].shape == (2, 4)

    def test_type_counts_sum(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = head_type_classifier(model, seqs)
        total = sum(result["type_counts"].values())
        assert total == 8


# ─── Component Report ─────────────────────────────────────────────────────


class TestComponentReport:
    def test_returns_dict(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = component_report(model, seqs)
        assert "head_classifications" in result
        assert "layer_summary" in result
        assert "n_layers" in result
        assert "n_heads" in result

    def test_layer_summary_length(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = component_report(model, seqs)
        assert len(result["layer_summary"]) == 2

    def test_n_layers_correct(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = component_report(model, seqs)
        assert result["n_layers"] == 2
        assert result["n_heads"] == 4
