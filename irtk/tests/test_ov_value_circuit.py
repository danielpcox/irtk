"""Tests for ov_value_circuit module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.ov_value_circuit import (
    ov_eigenspectrum, ov_token_copying_score,
    ov_writing_direction, ov_composition_with_next_layer,
    ov_unembed_alignment,
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
    return jnp.array([1, 5, 10, 15, 20])


def test_ov_eigenspectrum_structure(model):
    result = ov_eigenspectrum(model, layer=0, head=0)
    assert result['effective_rank'] > 0
    assert result['spectral_norm'] > 0
    assert len(result['top_singular_values']) <= 5


def test_ov_eigenspectrum_fractions(model):
    result = ov_eigenspectrum(model, layer=0, head=0)
    for sv in result['top_singular_values']:
        assert 0 <= sv['fraction'] <= 1


def test_ov_token_copying_score_structure(model):
    result = ov_token_copying_score(model, layer=0, head=0, token_ids=[1, 5, 10])
    assert len(result['per_token']) == 3
    assert isinstance(result['is_copy_head'], bool)


def test_ov_writing_direction_structure(model, tokens):
    result = ov_writing_direction(model, tokens, layer=0, head=0)
    assert len(result['per_source']) == 5
    assert result['mean_output_norm'] > 0


def test_ov_writing_direction_cosine_range(model, tokens):
    result = ov_writing_direction(model, tokens, layer=0, head=0)
    for s in result['per_source']:
        assert -1.0 <= s['cosine_with_mean'] <= 1.0


def test_ov_composition_structure(model):
    result = ov_composition_with_next_layer(model, layer=0, head=0)
    assert len(result['compositions']) == 4
    for c in result['compositions']:
        assert c['ov_to_q_norm'] >= 0


def test_ov_composition_last_layer(model):
    result = ov_composition_with_next_layer(model, layer=1, head=0)
    assert result['error'] == 'last_layer'


def test_ov_unembed_alignment_structure(model):
    result = ov_unembed_alignment(model, layer=0, head=0, top_k=5)
    assert len(result['promoted_tokens']) == 5
    assert len(result['suppressed_tokens']) == 5
    assert result['top_sv'] > 0


def test_ov_unembed_alignment_ordering(model):
    result = ov_unembed_alignment(model, layer=0, head=0, top_k=5)
    promoted_logits = [t['logit'] for t in result['promoted_tokens']]
    assert promoted_logits == sorted(promoted_logits, reverse=True)
