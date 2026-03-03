"""Tests for batch experiment utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.experiments import (
    ExperimentResult,
    run_on_dataset,
    sweep_ablations,
    find_circuit,
    compare_metrics,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


def _metric(logits):
    return float(logits[-1, 5])


class TestExperimentResult:
    def test_summary(self):
        result = ExperimentResult(
            values=np.array([1.0, 2.0, 3.0]),
            metadata={"method": "zero", "n_prompts": 3},
        )
        s = result.summary()
        assert "method: zero" in s
        assert "mean:" in s


class TestRunOnDataset:
    def test_mean_aggregate(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        result = run_on_dataset(model, seqs, _metric, aggregate="mean")
        assert result.values.ndim == 0  # scalar
        assert result.per_prompt is not None
        assert len(result.per_prompt) == 2

    def test_all_aggregate(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        result = run_on_dataset(model, seqs, _metric, aggregate="all")
        assert result.values.shape == (2,)

    def test_median_aggregate(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5]), jnp.array([6, 7, 8])]
        result = run_on_dataset(model, seqs, _metric, aggregate="median")
        assert result.values.ndim == 0

    def test_invalid_aggregate(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        with pytest.raises(ValueError, match="Unknown aggregate"):
            run_on_dataset(model, seqs, _metric, aggregate="invalid")


class TestSweepAblations:
    def test_heads_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = sweep_ablations(model, seqs, _metric, component="heads")
        assert result.values.shape == (2, 4)  # n_layers, n_heads

    def test_layers_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = sweep_ablations(model, seqs, _metric, component="layers")
        assert result.values.shape == (2,)  # n_layers

    def test_per_prompt_stored(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        result = sweep_ablations(model, seqs, _metric, component="heads")
        assert result.per_prompt is not None
        assert len(result.per_prompt) == 2

    def test_invalid_component(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        with pytest.raises(ValueError, match="Unknown component"):
            sweep_ablations(model, seqs, _metric, component="invalid")


class TestFindCircuit:
    def test_returns_expected_keys(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = find_circuit(model, seqs, _metric)
        assert "circuit_heads" in result.values
        assert "ablation_effects" in result.values
        assert "baseline" in result.values
        assert "relative_change" in result.values

    def test_ablation_effects_shape(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = find_circuit(model, seqs, _metric)
        assert result.values["ablation_effects"].shape == (2, 4)

    def test_circuit_heads_format(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2, 3])]
        result = find_circuit(model, seqs, _metric, threshold=0.0)
        # With threshold=0, all heads should be included (assuming non-zero baseline)
        for layer, head, importance in result.values["circuit_heads"]:
            assert 0 <= layer < 2
            assert 0 <= head < 4
            assert importance >= 0

    def test_metadata(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        result = find_circuit(model, seqs, _metric, threshold=0.5)
        assert result.metadata["threshold"] == 0.5
        assert result.metadata["n_prompts"] == 1


class TestCompareMetrics:
    def test_returns_all_metrics(self):
        model = _make_model()
        seqs = [jnp.array([0, 1, 2])]
        metrics = {
            "logit_5": lambda logits: float(logits[-1, 5]),
            "logit_10": lambda logits: float(logits[-1, 10]),
        }
        results = compare_metrics(model, seqs, metrics)
        assert "logit_5" in results
        assert "logit_10" in results
        assert isinstance(results["logit_5"], ExperimentResult)
