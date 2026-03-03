"""Tests for attention_head_importance module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.attention_head_importance import (
    head_knockout_importance,
    head_output_magnitude,
    head_logit_attribution,
    composite_importance_ranking,
    head_importance_by_position,
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


def test_head_knockout_importance(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_knockout_importance(model, tokens)
    assert len(result['per_head']) == 8
    assert result['most_important'] is not None
    # Should be sorted by importance
    for i in range(len(result['per_head']) - 1):
        assert result['per_head'][i]['max_logit_change'] >= result['per_head'][i+1]['max_logit_change'] - 0.01


def test_head_output_magnitude(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_magnitude(model, tokens)
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert h['output_norm'] >= 0


def test_head_logit_attribution(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_logit_attribution(model, tokens, position=-1)
    assert len(result['per_head']) == 8
    assert 0 <= result['target_token'] < 50
    # Should be sorted by absolute contribution
    for i in range(len(result['per_head']) - 1):
        assert result['per_head'][i]['abs_contribution'] >= result['per_head'][i+1]['abs_contribution'] - 0.01


def test_composite_importance_ranking(model_and_tokens):
    model, tokens = model_and_tokens
    result = composite_importance_ranking(model, tokens)
    assert len(result['per_head']) == 8
    assert result['most_important'] is not None
    for h in result['per_head']:
        assert h['composite_score'] >= 0


def test_head_importance_by_position(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_importance_by_position(model, tokens)
    assert len(result['per_position']) == 5
    for pos in result['per_position']:
        assert len(pos['head_scores']) == 8


def test_knockout_all_heads(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_knockout_importance(model, tokens)
    layers_heads = set()
    for h in result['per_head']:
        layers_heads.add((h['layer'], h['head']))
    assert len(layers_heads) == 8


def test_magnitude_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_output_magnitude(model, tokens)
    for i in range(len(result['per_head']) - 1):
        assert result['per_head'][i]['output_norm'] >= result['per_head'][i+1]['output_norm'] - 0.01


def test_composite_components(model_and_tokens):
    model, tokens = model_and_tokens
    result = composite_importance_ranking(model, tokens)
    for h in result['per_head']:
        assert h['output_norm'] >= 0
        assert h['logit_attribution'] >= 0


def test_position_importance_top(model_and_tokens):
    model, tokens = model_and_tokens
    result = head_importance_by_position(model, tokens)
    for pos in result['per_position']:
        assert pos['top_head'] is not None
        assert pos['top_head']['output_norm'] >= 0
