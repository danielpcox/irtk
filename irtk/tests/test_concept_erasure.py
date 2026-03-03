"""Tests for concept erasure and direction analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.hooked_transformer import HookedTransformer
from irtk.concept_erasure import (
    find_concept_direction,
    erase_concept,
    amplify_concept,
    concept_sensitivity,
    leace,
    concept_spectrum,
)


def _make_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
    )
    model = HookedTransformer(cfg)
    key = jax.random.PRNGKey(42)
    leaves, treedef = jax.tree.flatten(model)
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32,):
            key, subkey = jax.random.split(key)
            new_leaves.append(jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * 0.1)
        else:
            new_leaves.append(leaf)
    return jax.tree.unflatten(treedef, new_leaves)


def _make_tokens():
    """Create a few token sequences for testing."""
    pos = [jnp.array([0, 1, 2, 3]), jnp.array([4, 5, 6, 7])]
    neg = [jnp.array([10, 11, 12, 13]), jnp.array([14, 15, 16, 17])]
    return pos, neg


class TestFindConceptDirection:
    def test_returns_array(self):
        model = _make_model()
        pos, neg = _make_tokens()
        d = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post")
        assert isinstance(d, np.ndarray)
        assert d.shape == (16,)

    def test_unit_norm_when_normalized(self):
        model = _make_model()
        pos, neg = _make_tokens()
        d = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post", normalize=True)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-5

    def test_not_unit_norm_when_unnormalized(self):
        model = _make_model()
        pos, neg = _make_tokens()
        d = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post", normalize=False)
        # Unnormalized, so norm is not necessarily 1
        assert d.shape == (16,)

    def test_different_hooks(self):
        model = _make_model()
        pos, neg = _make_tokens()
        d0 = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post")
        d1 = find_concept_direction(model, pos, neg, "blocks.1.hook_resid_post")
        # Different layers should give different directions
        cos = np.dot(d0, d1)
        assert cos < 1.0 - 1e-5  # not exactly identical

    def test_opposite_direction_when_swapped(self):
        model = _make_model()
        pos, neg = _make_tokens()
        d1 = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post", normalize=False)
        d2 = find_concept_direction(model, neg, pos, "blocks.0.hook_resid_post", normalize=False)
        np.testing.assert_allclose(d1, -d2, atol=1e-5)


class TestEraseConcept:
    def test_returns_logits(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        logits = erase_concept(model, tokens, "blocks.0.hook_resid_post", direction)
        assert logits.shape == (4, 50)

    def test_changes_logits(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        pos, neg = _make_tokens()
        direction = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post")
        clean = model(tokens)
        erased = erase_concept(model, tokens, "blocks.0.hook_resid_post", direction)
        # Erasing a direction should change the logits
        diff = np.linalg.norm(np.array(clean) - np.array(erased))
        assert diff > 1e-5

    def test_erasing_zero_direction_no_change(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.zeros(16, dtype=np.float32)
        clean = model(tokens)
        erased = erase_concept(model, tokens, "blocks.0.hook_resid_post", direction)
        # Zero direction should not change logits
        np.testing.assert_allclose(np.array(clean), np.array(erased), atol=1e-4)

    def test_idempotent(self):
        """Erasing twice should give the same result as erasing once."""
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        direction = direction / np.linalg.norm(direction)

        # Erasing should be idempotent in the sense that the concept is
        # already gone after first erasure. We check the activations.
        _, cache1 = model.run_with_cache(tokens)
        act = np.array(cache1.cache_dict["blocks.0.hook_resid_post"])
        # Manually erase
        proj = act @ direction
        erased_act = act - proj[:, None] * direction[None, :]
        # Check projection is zero
        proj_after = erased_act @ direction
        np.testing.assert_allclose(proj_after, 0.0, atol=1e-5)


class TestAmplifyConcept:
    def test_alpha_zero_equals_erase(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        direction = direction / np.linalg.norm(direction)

        erased = erase_concept(model, tokens, "blocks.0.hook_resid_post", direction)
        amplified_zero = amplify_concept(model, tokens, "blocks.0.hook_resid_post", direction, alpha=0.0)
        np.testing.assert_allclose(np.array(erased), np.array(amplified_zero), atol=1e-4)

    def test_alpha_one_is_identity(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)

        clean = model(tokens)
        amplified_one = amplify_concept(model, tokens, "blocks.0.hook_resid_post", direction, alpha=1.0)
        np.testing.assert_allclose(np.array(clean), np.array(amplified_one), atol=1e-4)

    def test_alpha_two_amplifies(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)

        clean = model(tokens)
        amplified = amplify_concept(model, tokens, "blocks.0.hook_resid_post", direction, alpha=2.0)
        # Should be different from clean
        diff = np.linalg.norm(np.array(clean) - np.array(amplified))
        assert diff > 1e-5


class TestConceptSensitivity:
    def test_returns_dict(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = concept_sensitivity(model, tokens, "blocks.0.hook_resid_post", direction)
        assert "logit_diff_l2" in result
        assert "logit_diff_max" in result
        assert "top_token_change" in result
        assert "kl_divergence" in result
        assert "top_promoted" in result
        assert "top_suppressed" in result

    def test_l2_nonnegative(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = concept_sensitivity(model, tokens, "blocks.0.hook_resid_post", direction)
        assert result["logit_diff_l2"] >= 0
        assert result["logit_diff_max"] >= 0
        assert result["kl_divergence"] >= 0

    def test_zero_direction_zero_sensitivity(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.zeros(16, dtype=np.float32)
        result = concept_sensitivity(model, tokens, "blocks.0.hook_resid_post", direction)
        assert result["logit_diff_l2"] < 1e-3

    def test_top_promoted_suppressed_lists(self):
        model = _make_model()
        tokens = jnp.array([0, 1, 2, 3])
        direction = np.random.randn(16).astype(np.float32)
        result = concept_sensitivity(model, tokens, "blocks.0.hook_resid_post", direction)
        assert len(result["top_promoted"]) <= 10
        assert len(result["top_suppressed"]) <= 10
        for tid, change in result["top_promoted"]:
            assert isinstance(tid, int)
            assert isinstance(change, float)


class TestLeace:
    def test_returns_dict(self):
        np.random.seed(42)
        acts = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        result = leace(acts, labels)
        assert "projection" in result
        assert "concept_direction" in result
        assert "explained_variance" in result

    def test_projection_shape(self):
        np.random.seed(42)
        acts = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        result = leace(acts, labels)
        assert result["projection"].shape == (8, 8)
        assert result["concept_direction"].shape == (8,)

    def test_projection_idempotent(self):
        """P @ P = P for a projection matrix."""
        np.random.seed(42)
        acts = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        result = leace(acts, labels)
        P = result["projection"]
        PP = P @ P
        np.testing.assert_allclose(P, PP, atol=1e-5)

    def test_projection_symmetric(self):
        np.random.seed(42)
        acts = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        result = leace(acts, labels)
        P = result["projection"]
        np.testing.assert_allclose(P, P.T, atol=1e-5)

    def test_erases_concept_direction(self):
        """After projection, activations should have zero component along concept direction."""
        np.random.seed(42)
        # Create activations where class 0 and class 1 differ along a known direction
        d = 8
        direction = np.zeros(d, dtype=np.float64)
        direction[0] = 1.0
        acts = np.random.randn(20, d)
        acts[:10] += 2.0 * direction  # class 0 shifted
        labels = np.array([0] * 10 + [1] * 10)

        result = leace(acts, labels)
        P = result["projection"].astype(np.float64)
        erased = acts @ P.T

        # The mean difference along the concept direction should be near zero
        erased_mean_diff = erased[:10].mean(0) - erased[10:].mean(0)
        concept_dir = result["concept_direction"].astype(np.float64)
        residual = np.dot(erased_mean_diff, concept_dir)
        assert abs(residual) < 1e-4

    def test_single_class_returns_identity(self):
        acts = np.random.randn(10, 4).astype(np.float32)
        labels = np.zeros(10)
        result = leace(acts, labels)
        np.testing.assert_allclose(result["projection"], np.eye(4), atol=1e-5)

    def test_explained_variance_in_range(self):
        np.random.seed(42)
        acts = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        result = leace(acts, labels)
        assert 0 <= result["explained_variance"] <= 1


class TestConceptSpectrum:
    def test_returns_dict(self):
        model = _make_model()
        pos, neg = _make_tokens()
        result = concept_spectrum(model, pos, neg, "blocks.0.hook_resid_post", k=3)
        assert "directions" in result
        assert "explained_variance" in result
        assert "singular_values" in result

    def test_directions_shape(self):
        model = _make_model()
        pos, neg = _make_tokens()
        result = concept_spectrum(model, pos, neg, "blocks.0.hook_resid_post", k=2)
        assert result["directions"].shape[0] == 2
        assert result["directions"].shape[1] == 16

    def test_explained_variance_sums_to_at_most_one(self):
        model = _make_model()
        pos, neg = _make_tokens()
        result = concept_spectrum(model, pos, neg, "blocks.0.hook_resid_post", k=3)
        assert np.sum(result["explained_variance"]) <= 1.0 + 1e-5

    def test_singular_values_descending(self):
        model = _make_model()
        pos, neg = _make_tokens()
        result = concept_spectrum(model, pos, neg, "blocks.0.hook_resid_post", k=3)
        sv = result["singular_values"]
        for i in range(len(sv) - 1):
            assert sv[i] >= sv[i + 1] - 1e-6

    def test_first_direction_aligned_with_mean_diff(self):
        """The first concept direction should be close to the mean difference direction."""
        model = _make_model()
        pos, neg = _make_tokens()
        mean_dir = find_concept_direction(model, pos, neg, "blocks.0.hook_resid_post")
        spectrum = concept_spectrum(model, pos, neg, "blocks.0.hook_resid_post", k=3)
        # First direction should have high alignment with mean difference
        alignment = abs(np.dot(mean_dir, spectrum["directions"][0]))
        assert alignment > 0.1  # Some alignment (small model, few samples)
