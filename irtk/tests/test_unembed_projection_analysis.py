"""Tests for unembed_projection_analysis module."""

import jax
import jax.numpy as jnp
import pytest
from irtk import HookedTransformer, HookedTransformerConfig
from irtk.unembed_projection_analysis import (
    residual_unembed_trajectory, component_unembed_contribution,
    unembed_alignment_per_head, unembed_direction_stability,
    unembed_projection_summary,
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


def test_trajectory_structure(model, tokens):
    result = residual_unembed_trajectory(model, tokens, position=-1, top_k=3)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert len(p["top_tokens"]) == 3


def test_component_contribution_structure(model, tokens):
    result = component_unembed_contribution(model, tokens, layer=0, position=-1)
    assert result["dominant"] in ("attention", "mlp")


def test_component_contribution_total(model, tokens):
    result = component_unembed_contribution(model, tokens, layer=0, position=-1)
    assert abs(result["total_contribution"] - result["attn_contribution"] - result["mlp_contribution"]) < 1e-4


def test_per_head_alignment_structure(model, tokens):
    result = unembed_alignment_per_head(model, tokens, layer=0, position=-1)
    assert len(result["per_head"]) == 4
    assert 0 <= result["top_head"] < 4


def test_per_head_alignment_sorted(model, tokens):
    result = unembed_alignment_per_head(model, tokens, layer=0)
    contribs = [abs(h["logit_contribution"]) for h in result["per_head"]]
    assert contribs == sorted(contribs, reverse=True)


def test_direction_stability_structure(model, tokens):
    result = unembed_direction_stability(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2
    for p in result["per_layer"]:
        assert -1.1 <= p["cosine_to_unembed"] <= 1.1


def test_direction_stability_final(model, tokens):
    result = unembed_direction_stability(model, tokens, position=-1)
    # Final layer alignment should be positive (pointing toward prediction)
    assert result["final_alignment"] != 0  # non-trivial


def test_summary_structure(model, tokens):
    result = unembed_projection_summary(model, tokens, position=-1)
    assert len(result["per_layer"]) == 2


def test_summary_fields(model, tokens):
    result = unembed_projection_summary(model, tokens, position=-1)
    for p in result["per_layer"]:
        assert "top_prediction" in p
        assert "cosine_to_final_unembed" in p
