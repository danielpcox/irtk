"""Tests for knowledge_conflicts module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.knowledge_conflicts import (
    logit_direction_conflicts,
    residual_tug_of_war,
    attention_competition,
    interference_localization,
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


def test_logit_direction_conflicts(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_direction_conflicts(model, tokens)
    assert 'n_conflicts' in result
    assert 'top_conflicts' in result
    assert 'component_preferences' in result
    for pref in result['component_preferences']:
        assert 'name' in pref
        assert 'top_token' in pref
        assert isinstance(pref['top_token'], int)


def test_logit_conflicts_top_k(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_direction_conflicts(model, tokens, top_k=3)
    assert len(result['top_conflicts']) <= 3
    for conflict in result['top_conflicts']:
        assert conflict['promotes_a'] != conflict['promotes_b']


def test_residual_tug_of_war(model_and_tokens):
    model, tokens = model_and_tokens
    target = int(jnp.argmax(model(tokens)[-1]))
    result = residual_tug_of_war(model, tokens, target_token=target)
    assert result['target_token'] == target
    assert 'promoters' in result
    assert 'suppressors' in result
    assert isinstance(result['net_logit'], float)
    # Promoters have positive contribution
    for p in result['promoters']:
        assert p['contribution'] > 0
    # Suppressors have negative contribution
    for s in result['suppressors']:
        assert s['contribution'] < 0


def test_tug_of_war_components(model_and_tokens):
    model, tokens = model_and_tokens
    result = residual_tug_of_war(model, tokens, target_token=0)
    assert 'all_components' in result
    n_expected = 1 + 2 * model.cfg.n_layers  # embed + (attn + mlp) per layer
    assert len(result['all_components']) == n_expected


def test_attention_competition(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_competition(model, tokens)
    assert 'per_head' in result
    assert len(result['per_head']) == model.cfg.n_layers * model.cfg.n_heads
    for h in result['per_head']:
        assert 'competition_ratio' in h
        assert 0 <= h['competition_ratio'] <= 1.0
        assert 'attention_entropy' in h


def test_attention_competition_query_pos(model_and_tokens):
    model, tokens = model_and_tokens
    result = attention_competition(model, tokens, query_pos=2)
    assert result['query_position'] == 2
    assert result['most_competitive'] is not None


def test_interference_localization(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    result = interference_localization(model, tokens, tokens_b)
    assert 'per_component' in result
    assert 'top_divergent' in result
    assert 'residual_divergence' in result
    assert len(result['residual_divergence']) == model.cfg.n_layers
    for comp in result['per_component']:
        assert 'divergence' in comp
        assert comp['divergence'] >= 0
        assert -1 <= comp['cosine_similarity'] <= 1


def test_interference_top_k(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    result = interference_localization(model, tokens, tokens_b, top_k=2)
    assert len(result['top_divergent']) <= 2


def test_interference_sorted_by_divergence(model_and_tokens):
    model, tokens = model_and_tokens
    tokens_b = jnp.array([5, 15, 25, 35, 45])
    result = interference_localization(model, tokens, tokens_b)
    divs = [c['divergence'] for c in result['per_component']]
    assert divs == sorted(divs, reverse=True)
