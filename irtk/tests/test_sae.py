"""Tests for sparse autoencoders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from irtk.sae import (
    SparseAutoencoder,
    train_sae,
    feature_activation_stats,
    top_activating_examples,
    feature_logit_attribution,
)


class TestSparseAutoencoder:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        x = jnp.ones((10, 16))
        x_hat, feat_acts = sae(x)
        assert x_hat.shape == (10, 16)
        assert feat_acts.shape == (10, 64)

    def test_encode_shape(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        x = jnp.ones((5, 16))
        acts = sae.encode(x)
        assert acts.shape == (5, 64)
        # ReLU means non-negative
        assert jnp.all(acts >= 0)

    def test_decode_shape(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        acts = jnp.ones((5, 64))
        x_hat = sae.decode(acts)
        assert x_hat.shape == (5, 16)

    def test_feature_dirs(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        dirs = sae.feature_dirs()
        assert dirs.shape == (64, 16)

    def test_decoder_normalized(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        norms = jnp.linalg.norm(sae.W_dec, axis=-1)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-5)

    def test_top_features_1d(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        x = jax.random.normal(key, (16,))
        indices, acts = sae.top_features(x, k=5)
        assert indices.shape == (5,)
        assert acts.shape == (5,)
        # Should be sorted descending
        assert jnp.all(acts[:-1] >= acts[1:])

    def test_top_features_2d(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(16, 64, key=key)
        x = jax.random.normal(key, (3, 16))
        indices, acts = sae.top_features(x, k=5)
        assert indices.shape == (3, 5)
        assert acts.shape == (3, 5)


class TestTrainSAE:
    def test_basic(self):
        key = jax.random.PRNGKey(42)
        activations = jax.random.normal(key, (200, 8))
        result = train_sae(
            activations, d_model=8, n_features=32,
            epochs=3, batch_size=64, verbose=False,
        )
        assert result.sae is not None
        assert len(result.train_losses) == 3
        assert len(result.recon_losses) == 3
        assert len(result.l1_losses) == 3
        assert len(result.l0_sparsities) == 3
        assert len(result.val_losses) == 3

    def test_reconstruction_improves(self):
        key = jax.random.PRNGKey(42)
        activations = jax.random.normal(key, (500, 8))
        result = train_sae(
            activations, d_model=8, n_features=32,
            epochs=20, lr=1e-3, l1_coeff=1e-4,
            batch_size=128, verbose=False,
        )
        # Reconstruction loss should decrease
        assert result.recon_losses[-1] < result.recon_losses[0]

    def test_decoder_stays_normalized(self):
        key = jax.random.PRNGKey(42)
        activations = jax.random.normal(key, (100, 8))
        result = train_sae(
            activations, d_model=8, n_features=16,
            epochs=5, verbose=False,
        )
        norms = jnp.linalg.norm(result.sae.W_dec, axis=-1)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-4)


class TestFeatureAnalysis:
    def test_activation_stats(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(8, 32, key=key)
        acts = jax.random.normal(key, (50, 8))
        stats = feature_activation_stats(sae, acts)
        assert stats["mean_acts"].shape == (32,)
        assert stats["firing_rate"].shape == (32,)
        assert stats["max_acts"].shape == (32,)
        assert 0 <= stats["l0_mean"]

    def test_top_activating_examples(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(8, 32, key=key)
        acts = jax.random.normal(key, (50, 8))
        indices, values = top_activating_examples(sae, 0, acts, k=5)
        assert len(indices) == 5
        assert len(values) == 5
        # Should be sorted descending
        assert np.all(values[:-1] >= values[1:])

    def test_feature_logit_attribution(self):
        key = jax.random.PRNGKey(0)
        sae = SparseAutoencoder(8, 32, key=key)
        W_U = jax.random.normal(key, (8, 100))
        promoted, suppressed = feature_logit_attribution(sae, W_U, 0, k=5)
        assert len(promoted) == 5
        assert len(suppressed) == 5
