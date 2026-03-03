"""Tests for prediction_decomposition module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.prediction_decomposition import (
    logit_buildup,
    attn_vs_mlp_logit_share,
    confidence_evolution,
    alternative_predictions,
    embedding_logit_bias,
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
    tokens = jnp.array([1, 10, 20, 30, 40])
    return model, tokens


def test_logit_buildup(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_buildup(model, tokens)
    assert 'target_token' in result
    assert len(result['steps']) == 3  # embedding + 2 layers
    assert result['steps'][0]['stage'] == 'embedding'


def test_logit_buildup_target(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_buildup(model, tokens, target_token=5)
    assert result['target_token'] == 5


def test_attn_vs_mlp_logit_share(model_and_tokens):
    model, tokens = model_and_tokens
    result = attn_vs_mlp_logit_share(model, tokens, top_k=3)
    assert len(result['per_token']) == 3
    for t in result['per_token']:
        assert 0 <= t['attn_share'] <= 1.0
        assert 0 <= t['mlp_share'] <= 1.0


def test_confidence_evolution(model_and_tokens):
    model, tokens = model_and_tokens
    result = confidence_evolution(model, tokens)
    assert len(result['per_layer']) == 3  # embedding + 2 layers
    for p in result['per_layer']:
        assert 0 <= p['confidence'] <= 1.0
        assert p['entropy'] >= 0


def test_confidence_has_commit(model_and_tokens):
    model, tokens = model_and_tokens
    result = confidence_evolution(model, tokens)
    assert 'commit_stage' in result
    assert 'final_token' in result


def test_alternative_predictions(model_and_tokens):
    model, tokens = model_and_tokens
    result = alternative_predictions(model, tokens, top_k=3)
    assert len(result['final_top_tokens']) == 3
    assert len(result['per_layer']) == 2  # 2 layers
    for p in result['per_layer']:
        assert len(p['rankings']) == 3


def test_alternative_ranks_valid(model_and_tokens):
    model, tokens = model_and_tokens
    result = alternative_predictions(model, tokens, top_k=3)
    # All ranks should be valid non-negative integers
    for p in result['per_layer']:
        for r in p['rankings']:
            assert r['rank'] >= 0


def test_embedding_logit_bias(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_logit_bias(model, tokens, top_k=5)
    assert len(result['embed_predictions']) == 5
    assert result['embed_rank_of_final'] >= 0
    for p in result['embed_predictions']:
        assert p['probability'] >= 0


def test_embedding_entropy(model_and_tokens):
    model, tokens = model_and_tokens
    result = embedding_logit_bias(model, tokens)
    assert result['embed_entropy'] >= 0
