"""Tests for attention_knowledge_retrieval module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_knowledge_retrieval import (
    query_key_matching,
    value_extraction_pattern,
    knowledge_routing,
    retrieval_vs_computation,
    factual_association_strength,
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


class TestQueryKeyMatching:
    def test_basic(self, model, tokens):
        result = query_key_matching(model, tokens, layer=0, head=0)
        assert "top_matches" in result
        assert "selectivity" in result
        assert result["query_norm"] > 0

    def test_top_matches(self, model, tokens):
        result = query_key_matching(model, tokens, layer=0, head=0, top_k=3)
        assert len(result["top_matches"]) == 3


class TestValueExtractionPattern:
    def test_basic(self, model, tokens):
        result = value_extraction_pattern(model, tokens, layer=0, head=0)
        assert "per_source" in result
        assert "dominant_source" in result
        assert "value_diversity" in result

    def test_output_norm(self, model, tokens):
        result = value_extraction_pattern(model, tokens, layer=0, head=0)
        assert result["total_output_norm"] > 0


class TestKnowledgeRouting:
    def test_basic(self, model, tokens):
        result = knowledge_routing(model, tokens)
        assert "routing_matrix" in result
        assert "top_routing_heads" in result

    def test_matrix_shape(self, model, tokens):
        result = knowledge_routing(model, tokens)
        assert result["routing_matrix"].shape == (model.cfg.n_layers, model.cfg.n_heads)


class TestRetrievalVsComputation:
    def test_basic(self, model, tokens):
        result = retrieval_vs_computation(model, tokens, layer=0)
        assert "per_head" in result
        assert len(result["per_head"]) == model.cfg.n_heads

    def test_classification(self, model, tokens):
        result = retrieval_vs_computation(model, tokens, layer=0)
        for h in result["per_head"]:
            assert h["classification"] in ("retrieval", "computation")


class TestFactualAssociationStrength:
    def test_basic(self, model, tokens):
        result = factual_association_strength(model, tokens)
        assert "per_head" in result
        assert "strongest_associations" in result
        assert "aggregate_strength" in result

    def test_associations_populated(self, model, tokens):
        result = factual_association_strength(model, tokens)
        assert len(result["per_head"]) > 0
