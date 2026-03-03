"""Tests for gradient-based interpretability utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.gradients import (
    gradient_x_input,
    gradient_norm,
    integrated_gradients,
    integrated_gradients_full,
    logit_gradient_attribution,
    input_jacobian,
    token_attribution,
    _embed_to_logits,
    _get_embeddings,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    return HookedTransformer(cfg)


def _make_model_random():
    """Create a model with random weights so gradients are non-trivial."""
    import equinox as eqx
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    # Replace zero-init arrays with random ones
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


class TestEmbedToLogits:
    def test_matches_full_forward(self):
        """_embed_to_logits should match model(tokens) when given the same embeddings."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3, 4])
        logits_direct = model(tokens)

        embeddings = _get_embeddings(model, tokens)
        logits_via_embed = _embed_to_logits(model, embeddings)

        np.testing.assert_allclose(
            np.array(logits_direct), np.array(logits_via_embed), atol=1e-5
        )

    def test_embeddings_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        embed = _get_embeddings(model, tokens)
        assert embed.shape == (3, 16)


class TestGradientXInput:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attr = gradient_x_input(model, tokens, target_token=5)
        assert attr.shape == (4,)

    def test_nonnegative(self):
        """Norm-based attribution should be non-negative."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attr = gradient_x_input(model, tokens, target_token=5)
        assert np.all(attr >= 0)

    def test_different_targets_differ(self):
        """Different target tokens should give different attributions."""
        model = _make_model_random()
        tokens = jnp.array([0, 1, 2, 3])
        attr1 = gradient_x_input(model, tokens, target_token=5)
        attr2 = gradient_x_input(model, tokens, target_token=10)
        assert not np.allclose(attr1, attr2)


class TestGradientNorm:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([1, 2, 3])
        attr = gradient_norm(model, tokens, target_token=5)
        assert attr.shape == (3,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([1, 2, 3])
        attr = gradient_norm(model, tokens, target_token=5)
        assert np.all(attr >= 0)


class TestIntegratedGradients:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attr = integrated_gradients(model, tokens, target_token=5, n_steps=10)
        assert attr.shape == (4,)

    def test_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        attr = integrated_gradients(model, tokens, target_token=5, n_steps=10)
        assert np.all(attr >= 0)

    def test_custom_baseline(self):
        """Should accept a custom baseline embedding."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        embeddings = _get_embeddings(model, tokens)
        baseline = jnp.ones_like(embeddings) * 0.1
        attr = integrated_gradients(model, tokens, target_token=5, n_steps=10, baseline=baseline)
        assert attr.shape == (3,)

    def test_more_steps_changes_result(self):
        """Different step counts should give slightly different results."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        attr5 = integrated_gradients(model, tokens, target_token=5, n_steps=5)
        attr50 = integrated_gradients(model, tokens, target_token=5, n_steps=50)
        # They should be close but not identical
        assert attr5.shape == attr50.shape


class TestIntegratedGradientsFull:
    def test_output_shape(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        attr = integrated_gradients_full(model, tokens, target_token=5, n_steps=10)
        assert attr.shape == (3, 16)  # [seq_len, d_model]


class TestLogitGradientAttribution:
    def test_returns_all_components(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        results = logit_gradient_attribution(model, tokens, target_token=5)

        assert "embed" in results
        for l in range(model.cfg.n_layers):
            assert f"L{l}_attn" in results
            assert f"L{l}_mlp" in results

    def test_attributions_are_scalars(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        results = logit_gradient_attribution(model, tokens, target_token=5)
        for name, value in results.items():
            assert np.isscalar(value) or value.ndim == 0, f"{name} is not scalar"


class TestInputJacobian:
    def test_shapes(self):
        """Jacobian should have correct shapes."""
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=16, d_head=4, n_heads=2, d_vocab=20,
        )
        model = HookedTransformer(cfg)
        tokens = jnp.array([0, 1, 2])

        jac, top_entries = input_jacobian(model, tokens, top_k=5)
        assert jac.shape == (20, 3, 8)  # [d_vocab, seq_len, d_model]
        assert len(top_entries) == 5
        for token_id, input_pos, norm_val in top_entries:
            assert 0 <= token_id < 20
            assert 0 <= input_pos < 3
            assert norm_val >= 0

    def test_top_entries_sorted(self):
        """Top entries should be in descending order of norm."""
        cfg = HookedTransformerConfig(
            n_layers=1, d_model=8, n_ctx=16, d_head=4, n_heads=2, d_vocab=20,
        )
        model = HookedTransformer(cfg)
        tokens = jnp.array([0, 1, 2])

        _, top_entries = input_jacobian(model, tokens, top_k=5)
        norms = [e[2] for e in top_entries]
        assert norms == sorted(norms, reverse=True)


class TestTokenAttribution:
    def test_grad_x_input(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        attr = token_attribution(model, tokens, target_token=5, method="grad_x_input")
        assert attr.shape == (3,)

    def test_grad_norm(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        attr = token_attribution(model, tokens, target_token=5, method="grad_norm")
        assert attr.shape == (3,)

    def test_integrated_gradients(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        attr = token_attribution(model, tokens, target_token=5, method="integrated_gradients", n_steps=5)
        assert attr.shape == (3,)

    def test_invalid_method(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match="Unknown attribution method"):
            token_attribution(model, tokens, target_token=5, method="invalid")
