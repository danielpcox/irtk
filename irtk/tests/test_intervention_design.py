"""Tests for intervention_design module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.intervention_design import (
    scale_component_effect, add_direction_effect, zero_ablation_sweep,
    mean_ablation_effect, progressive_ablation,
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


def test_scale_component_effect_basic(model_and_tokens):
    model, tokens = model_and_tokens
    result = scale_component_effect(model, tokens, 'L0_attn', 0.0)
    assert 'component' in result
    assert result['component'] == 'L0_attn'
    assert result['scale'] == 0.0
    assert 'clean_prediction' in result
    assert 'kl_divergence' in result
    assert result['kl_divergence'] >= 0


def test_scale_component_effect_mlp(model_and_tokens):
    model, tokens = model_and_tokens
    result = scale_component_effect(model, tokens, 'L1_mlp', 2.0)
    assert result['component'] == 'L1_mlp'
    assert result['scale'] == 2.0
    assert isinstance(result['prediction_changed'], bool)


def test_add_direction_effect(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(0), (16,))
    result = add_direction_effect(model, tokens, layer=0, direction=direction, magnitude=5.0)
    assert 'layer' in result
    assert result['layer'] == 0
    assert result['magnitude'] == 5.0
    assert 'clean_prediction' in result
    assert 'steered_prediction' in result
    assert isinstance(result['prediction_changed'], bool)


def test_add_direction_effect_position(model_and_tokens):
    model, tokens = model_and_tokens
    direction = jax.random.normal(jax.random.PRNGKey(1), (16,))
    result = add_direction_effect(model, tokens, layer=1, direction=direction, position=2)
    assert result['position'] == 2


def test_zero_ablation_sweep(model_and_tokens):
    model, tokens = model_and_tokens
    result = zero_ablation_sweep(model, tokens)
    assert 'per_component' in result
    assert len(result['per_component']) == 4  # 2 layers * 2 components
    assert 'most_important' in result
    for comp in result['per_component']:
        assert 'component' in comp
        assert 'kl_divergence' in comp
        assert comp['kl_divergence'] >= 0


def test_zero_ablation_sweep_sorted(model_and_tokens):
    model, tokens = model_and_tokens
    result = zero_ablation_sweep(model, tokens)
    kls = [c['kl_divergence'] for c in result['per_component']]
    assert kls == sorted(kls, reverse=True)


def test_mean_ablation_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_ablation_effect(model, tokens, layer=0, component='attn')
    assert result['layer'] == 0
    assert result['component'] == 'attn'
    assert 'kl_divergence' in result
    assert result['kl_divergence'] >= 0
    assert isinstance(result['prediction_changed'], bool)


def test_mean_ablation_effect_mlp(model_and_tokens):
    model, tokens = model_and_tokens
    result = mean_ablation_effect(model, tokens, layer=1, component='mlp')
    assert result['component'] == 'mlp'


def test_progressive_ablation(model_and_tokens):
    model, tokens = model_and_tokens
    result = progressive_ablation(model, tokens)
    assert 'per_step' in result
    assert len(result['per_step']) == 2  # 2 layers
    assert 'clean_prediction' in result
    assert 'min_layers_needed' in result
    for step in result['per_step']:
        assert 'n_ablated' in step
        assert 'kl_divergence' in step
        assert step['kl_divergence'] >= 0
