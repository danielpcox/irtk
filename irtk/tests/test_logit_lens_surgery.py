"""Tests for logit_lens_surgery module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.logit_lens_surgery import (
    logit_lens_at_layer,
    logit_lens_diff,
    logit_lens_intervention,
    component_logit_lens_effect,
    logit_lens_trajectory,
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


def test_logit_lens_at_layer(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_at_layer(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    for p in result['per_position']:
        assert 0 <= p['top_token'] < 50
        assert p['confidence'] >= 0
        assert p['entropy'] >= 0


def test_logit_lens_diff(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_diff(model, tokens, layer_a=0, layer_b=1)
    assert len(result['per_position']) == 5
    assert result['mean_kl'] >= 0
    assert 0 <= result['change_fraction'] <= 1.0


def test_logit_lens_intervention(model_and_tokens):
    model, tokens = model_and_tokens

    def zero_attn(x, name):
        return jnp.zeros_like(x)

    result = logit_lens_intervention(model, tokens, layer=1,
                                      hook_name='blocks.0.hook_attn_out',
                                      hook_fn=zero_attn)
    assert len(result['per_position']) == 5
    assert result['mean_max_change'] >= 0


def test_component_logit_lens_effect(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_logit_lens_effect(model, tokens, layer=0)
    assert len(result['per_position']) == 5
    assert result['attn_changes'] >= 0
    assert result['mlp_changes'] >= 0


def test_logit_lens_trajectory(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_trajectory(model, tokens, position=-1, top_k=3)
    assert len(result['stages']) == 2
    for s in result['stages']:
        assert len(s['top_predictions']) == 3
        assert s['entropy'] >= 0


def test_trajectory_top_k(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_trajectory(model, tokens, position=0, top_k=5)
    for s in result['stages']:
        assert len(s['top_predictions']) == 5


def test_diff_same_layer(model_and_tokens):
    model, tokens = model_and_tokens
    result = logit_lens_diff(model, tokens, layer_a=0, layer_b=0)
    assert result['n_changed'] == 0
    assert result['mean_kl'] < 0.01


def test_component_effect_per_pos(model_and_tokens):
    model, tokens = model_and_tokens
    result = component_logit_lens_effect(model, tokens, layer=0)
    for p in result['per_position']:
        assert p['attn_max_logit_change'] >= 0
        assert p['mlp_max_logit_change'] >= 0


def test_lens_at_all_layers(model_and_tokens):
    model, tokens = model_and_tokens
    for l in range(2):
        result = logit_lens_at_layer(model, tokens, layer=l)
        assert result['mean_confidence'] > 0
