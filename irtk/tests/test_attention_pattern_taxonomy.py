"""Tests for attention_pattern_taxonomy module."""
import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_pattern_taxonomy import (
    pattern_diagonal_score, pattern_uniformity_score,
    pattern_sparsity_score, pattern_locality_score,
    classify_attention_patterns, attention_pattern_taxonomy_summary,
)

@pytest.fixture
def model_and_tokens():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype == jnp.float32:
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape) * 0.1)
        else:
            new_leaves.append(leaf)
    model = jax.tree.unflatten(treedef, new_leaves)
    tokens = jnp.array([1, 5, 10, 15, 20])
    return model, tokens

def test_diagonal_score_perfect():
    # Perfect diagonal pattern (prev token)
    pat = jnp.zeros((4, 4))
    pat = pat.at[1, 0].set(1.0)
    pat = pat.at[2, 1].set(1.0)
    pat = pat.at[3, 2].set(1.0)
    pat = pat.at[0, 0].set(1.0)
    score = pattern_diagonal_score(pat)
    assert score > 0.9

def test_uniformity_score():
    # Uniform causal pattern
    pat = jnp.zeros((4, 4))
    for i in range(4):
        pat = pat.at[i, :i+1].set(1.0 / (i + 1))
    score = pattern_uniformity_score(pat)
    assert score > 0.5

def test_sparsity_score_range():
    pat = jnp.eye(5) * 0.8 + 0.04
    # Make causal
    pat = jnp.tril(pat)
    pat = pat / (jnp.sum(pat, axis=-1, keepdims=True) + 1e-10)
    score = pattern_sparsity_score(pat)
    assert 0 <= score <= 1.0

def test_locality_score():
    # Local pattern
    pat = jnp.zeros((5, 5))
    for i in range(5):
        pat = pat.at[i, max(0, i-1):i+1].set(0.5)
    pat = pat / (jnp.sum(pat, axis=-1, keepdims=True) + 1e-10)
    score = pattern_locality_score(pat, window=2)
    assert score > 0.5

def test_classify_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_attention_patterns(model, tokens, layer=0)
    assert "per_head" in result
    assert len(result["per_head"]) == 4

def test_classify_categories(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_attention_patterns(model, tokens, layer=0)
    valid = {"diagonal", "uniform", "sparse", "local", "mixed"}
    for h in result["per_head"]:
        assert h["category"] in valid
        assert "scores" in h

def test_classify_score_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = classify_attention_patterns(model, tokens, layer=0)
    for h in result["per_head"]:
        for score_name, score_val in h["scores"].items():
            assert isinstance(score_val, float)

def test_taxonomy_summary_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_taxonomy_summary(model, tokens)
    assert "per_layer" in result
    assert len(result["per_layer"]) == 2
    assert "overall_distribution" in result

def test_taxonomy_summary_counts(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_pattern_taxonomy_summary(model, tokens)
    total = sum(result["overall_distribution"].values())
    assert total == 8  # 2 layers * 4 heads
