"""Tests for logit formation analysis."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_formation_analysis import (
    logit_buildup_trajectory, top_logit_contributors,
    logit_competition_analysis, embedding_logit_bias,
    logit_formation_summary,
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


def test_logit_buildup_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_buildup_trajectory(model, tokens, position=-1, token_id=5)
    assert "per_layer" in result
    assert "final_logit" in result
    assert result["token_id"] == 5
    assert len(result["per_layer"]) == 2


def test_logit_buildup_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_buildup_trajectory(model, tokens, position=-1, token_id=5)
    for p in result["per_layer"]:
        assert "logit" in p
        assert "attn_contribution" in p
        assert "mlp_contribution" in p


def test_top_contributors_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = top_logit_contributors(model, tokens, position=-1, top_k=3)
    assert "top_token" in result
    assert "top_contributors" in result
    assert len(result["top_contributors"]) == 3


def test_top_contributors_ordered(model_and_tokens):
    model, tokens = model_and_tokens
    result = top_logit_contributors(model, tokens, position=-1, top_k=5)
    magnitudes = [abs(c["contribution"]) for c in result["top_contributors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_competition_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_competition_analysis(model, tokens, position=-1, top_k=3)
    assert "tracked_tokens" in result
    assert "per_layer" in result
    assert "leader_changes" in result
    assert len(result["tracked_tokens"]) == 3


def test_competition_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_competition_analysis(model, tokens, position=-1, top_k=3)
    for p in result["per_layer"]:
        assert "token_logits" in p
        assert "leader" in p
        assert p["leader"] in result["tracked_tokens"]


def test_embedding_bias_structure(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_logit_bias(model, tokens, position=-1, top_k=5)
    assert "top_predictions" in result
    assert "embed_norm" in result
    assert "pos_embed_norm" in result
    assert len(result["top_predictions"]) == 5


def test_embedding_bias_values(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_logit_bias(model, tokens, position=-1)
    assert result["embed_norm"] > 0
    assert result["pos_embed_norm"] > 0


def test_logit_formation_summary(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_formation_summary(model, tokens, position=-1)
    assert "top_token" in result
    assert "top_contributor" in result
    assert "leader_changes" in result
