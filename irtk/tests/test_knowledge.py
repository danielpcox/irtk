"""Tests for knowledge and factual recall analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.knowledge import (
    knowledge_neurons,
    causal_knowledge_tracing,
    fact_editing_vector,
    attribute_to_mlp_vs_attn,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
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


class TestKnowledgeNeurons:
    def test_returns_list(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knowledge_neurons(model, tokens, target_token=5, top_k=10)
        assert isinstance(result, list)
        assert len(result) == 10

    def test_dict_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knowledge_neurons(model, tokens, target_token=5, top_k=5)
        for entry in result:
            assert "layer" in entry
            assert "neuron" in entry
            assert "attribution" in entry
            assert "activation" in entry

    def test_sorted_by_magnitude(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knowledge_neurons(model, tokens, target_token=5, top_k=10)
        magnitudes = [abs(r["attribution"]) for r in result]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_valid_layer_neuron(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = knowledge_neurons(model, tokens, target_token=5, top_k=5)
        for entry in result:
            assert 0 <= entry["layer"] < model.cfg.n_layers
            assert entry["neuron"] >= 0


class TestCausalKnowledgeTracing:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = causal_knowledge_tracing(
            model, tokens, subject_pos=[1, 2], target_token=10
        )
        assert "clean_logit" in result
        assert "corrupted_logit" in result
        assert "restored_resid" in result
        assert "restored_mlp" in result
        assert "restored_attn" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = causal_knowledge_tracing(
            model, tokens, subject_pos=[1, 2], target_token=10
        )
        assert result["restored_resid"].shape == (model.cfg.n_layers,)
        assert result["restored_mlp"].shape == (model.cfg.n_layers,)
        assert result["restored_attn"].shape == (model.cfg.n_layers,)

    def test_clean_logit_is_scalar(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = causal_knowledge_tracing(
            model, tokens, subject_pos=[1], target_token=5
        )
        assert isinstance(result["clean_logit"], float)
        assert isinstance(result["corrupted_logit"], float)

    def test_corruption_changes_logit(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4, 5])
        result = causal_knowledge_tracing(
            model, tokens, subject_pos=[1, 2], target_token=10, noise_std=10.0
        )
        # Corruption should change the logit
        assert result["clean_logit"] != result["corrupted_logit"]


class TestFactEditingVector:
    def test_returns_array(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vec = fact_editing_vector(model, tokens, old_token=5, new_token=10, layer=0)
        assert vec.shape == (model.cfg.d_model,)

    def test_unit_norm(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vec = fact_editing_vector(model, tokens, old_token=5, new_token=10, layer=0)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_different_tokens_different_vectors(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        vec1 = fact_editing_vector(model, tokens, old_token=5, new_token=10, layer=0)
        vec2 = fact_editing_vector(model, tokens, old_token=5, new_token=20, layer=0)
        assert not np.allclose(vec1, vec2, atol=1e-3)


class TestAttributeToMlpVsAttn:
    def test_returns_expected_keys(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attribute_to_mlp_vs_attn(model, tokens, target_token=5)
        assert "mlp_contrib" in result
        assert "attn_contrib" in result
        assert "embed_contrib" in result
        assert "total_logit" in result

    def test_shapes(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attribute_to_mlp_vs_attn(model, tokens, target_token=5)
        assert result["mlp_contrib"].shape == (model.cfg.n_layers,)
        assert result["attn_contrib"].shape == (model.cfg.n_layers,)

    def test_scalar_values(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result = attribute_to_mlp_vs_attn(model, tokens, target_token=5)
        assert isinstance(result["embed_contrib"], float)
        assert isinstance(result["total_logit"], float)

    def test_pos_parameter(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        result_last = attribute_to_mlp_vs_attn(model, tokens, target_token=5, pos=-1)
        result_first = attribute_to_mlp_vs_attn(model, tokens, target_token=5, pos=0)
        # Different positions should give different attributions
        assert not np.allclose(result_last["mlp_contrib"], result_first["mlp_contrib"])
