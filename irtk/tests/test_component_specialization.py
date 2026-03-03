"""Tests for component_specialization module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.component_specialization import (
    head_function_profile,
    mlp_specialization,
    component_consistency,
    specialization_spectrum,
    component_redundancy,
)


@pytest.fixture
def model_and_data():
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
    tokens_list = [
        jnp.array([1, 10, 20, 30, 40]),
        jnp.array([5, 15, 25, 35, 45]),
        jnp.array([2, 12, 22, 32, 42]),
    ]
    return model, tokens, tokens_list


def test_head_function_profile(model_and_data):
    model, tokens, _ = model_and_data
    result = head_function_profile(model, tokens)
    assert 'per_head' in result
    assert len(result['per_head']) == 8
    for h in result['per_head']:
        assert 0 <= h['self_attention'] <= 1.0
        assert h['dominant_function'] in ['self', 'previous_token', 'first_token']
    assert 'function_distribution' in result


def test_mlp_specialization(model_and_data):
    model, _, tokens_list = model_and_data
    result = mlp_specialization(model, tokens_list, layer=0, top_k=5)
    assert result['layer'] == 0
    assert len(result['most_specialized']) <= 5
    assert len(result['least_specialized']) <= 5
    for n in result['most_specialized']:
        assert 0 <= n['specialization'] <= 1.0
        assert 0 <= n['activation_frequency'] <= 1.0


def test_mlp_specialization_layer1(model_and_data):
    model, _, tokens_list = model_and_data
    result = mlp_specialization(model, tokens_list, layer=1)
    assert result['layer'] == 1


def test_component_consistency(model_and_data):
    model, _, tokens_list = model_and_data
    result = component_consistency(model, tokens_list)
    assert 'per_component' in result
    assert len(result['per_component']) == 4  # 2 layers * (attn + mlp)
    for c in result['per_component']:
        assert 0 <= c['consistency'] <= 1.0
    assert result['most_consistent'] is not None


def test_specialization_spectrum(model_and_data):
    model, _, tokens_list = model_and_data
    result = specialization_spectrum(model, tokens_list)
    assert 'per_component' in result
    for c in result['per_component']:
        assert c['classification'] in ['specialist', 'generalist']
    total = result['n_specialists'] + result['n_generalists']
    assert total == len(result['per_component'])


def test_component_redundancy(model_and_data):
    model, tokens, _ = model_and_data
    result = component_redundancy(model, tokens)
    assert 'n_redundant_pairs' in result
    assert result['n_components'] == 4
    for pair in result['redundant_pairs']:
        assert abs(pair['cosine_similarity']) > 0.8


def test_head_profile_has_all_heads(model_and_data):
    model, tokens, _ = model_and_data
    result = head_function_profile(model, tokens)
    layers = set(h['layer'] for h in result['per_head'])
    heads = set(h['head'] for h in result['per_head'])
    assert layers == {0, 1}
    assert heads == {0, 1, 2, 3}


def test_consistency_sorted(model_and_data):
    model, _, tokens_list = model_and_data
    result = component_consistency(model, tokens_list)
    consistencies = [c['consistency'] for c in result['per_component']]
    assert consistencies == sorted(consistencies, reverse=True)


def test_spectrum_sorted(model_and_data):
    model, _, tokens_list = model_and_data
    result = specialization_spectrum(model, tokens_list)
    scores = [c['specialization_score'] for c in result['per_component']]
    assert scores == sorted(scores, reverse=True)
