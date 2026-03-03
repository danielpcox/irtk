"""Sparse Autoencoders (SAEs) for discovering interpretable features.

SAEs learn a sparse, overcomplete dictionary of features from model activations.
Each feature is a direction in activation space that fires on specific inputs.

Supports:
- Training SAEs on cached activations from any hook point
- Standard SAE architecture (encoder/decoder with ReLU + L1 sparsity)
- Feature analysis: top activating examples, feature attribution, etc.

References:
    - Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
    - Bricken et al. (2023) "Towards Monosemanticity" (Anthropic)
"""

from typing import Optional
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np


class SparseAutoencoder(eqx.Module):
    """Sparse autoencoder for decomposing activations into interpretable features.

    Architecture:
        encoder: x_centered @ W_enc + b_enc -> ReLU -> feature_acts
        decoder: feature_acts @ W_dec + b_dec -> x_reconstructed

    The decoder columns are the learned feature directions.
    W_dec is kept unit-norm per feature (row-normalized).
    """

    W_enc: jnp.ndarray   # [d_model, n_features]
    b_enc: jnp.ndarray   # [n_features]
    W_dec: jnp.ndarray   # [n_features, d_model]
    b_dec: jnp.ndarray   # [d_model]

    d_model: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_features: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.d_model = d_model
        self.n_features = n_features

        k1, k2 = jax.random.split(key)
        # Initialize encoder as transpose of decoder (tied init)
        self.W_dec = jax.random.normal(k1, (n_features, d_model)) * (1.0 / jnp.sqrt(d_model))
        # Normalize decoder rows
        self.W_dec = self.W_dec / jnp.linalg.norm(self.W_dec, axis=-1, keepdims=True)
        self.W_enc = self.W_dec.T.copy()  # [d_model, n_features]
        self.b_enc = jnp.zeros(n_features)
        self.b_dec = jnp.zeros(d_model)

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode activations to sparse feature activations.

        Args:
            x: [..., d_model] activations.

        Returns:
            [..., n_features] sparse feature activations (after ReLU).
        """
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        return jax.nn.relu(pre_acts)

    def decode(self, feature_acts: jnp.ndarray) -> jnp.ndarray:
        """Decode feature activations back to activation space.

        Args:
            feature_acts: [..., n_features] sparse feature activations.

        Returns:
            [..., d_model] reconstructed activations.
        """
        return feature_acts @ self.W_dec + self.b_dec

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Full forward pass: encode then decode.

        Args:
            x: [..., d_model] activations.

        Returns:
            (x_reconstructed, feature_acts) tuple.
        """
        feature_acts = self.encode(x)
        x_hat = self.decode(feature_acts)
        return x_hat, feature_acts

    def feature_dirs(self) -> jnp.ndarray:
        """Get the learned feature directions (decoder rows).

        Returns:
            [n_features, d_model] feature direction vectors.
        """
        return self.W_dec

    def top_features(
        self, x: jnp.ndarray, k: int = 10
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the top-k most active features for an input.

        Args:
            x: [d_model] single activation vector, or [seq_len, d_model].
            k: Number of top features to return.

        Returns:
            (indices, activations): top-k feature indices and their activation values.
            If x is 2D, returns [seq_len, k] arrays.
        """
        acts = self.encode(x)
        if acts.ndim == 1:
            top_idx = jnp.argsort(acts)[::-1][:k]
            return top_idx, acts[top_idx]
        else:
            top_idx = jnp.argsort(acts, axis=-1)[:, ::-1][:, :k]
            top_acts = jnp.take_along_axis(acts, top_idx, axis=-1)
            return top_idx, top_acts


def _normalize_decoder(sae: SparseAutoencoder) -> SparseAutoencoder:
    """Normalize decoder rows to unit norm."""
    norms = jnp.linalg.norm(sae.W_dec, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-8)
    new_W_dec = sae.W_dec / norms
    return eqx.tree_at(lambda s: s.W_dec, sae, new_W_dec)


@dataclass
class SAETrainResult:
    """Results from training a sparse autoencoder."""

    sae: SparseAutoencoder
    train_losses: list[float]
    recon_losses: list[float]
    l1_losses: list[float]
    val_losses: list[float]
    l0_sparsities: list[float]  # average number of active features


def train_sae(
    activations: jnp.ndarray,
    d_model: int,
    n_features: int,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    epochs: int = 10,
    batch_size: int = 256,
    val_frac: float = 0.1,
    normalize_decoder: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> SAETrainResult:
    """Train a sparse autoencoder on activations.

    Loss = MSE(x, x_hat) + l1_coeff * L1(feature_acts)

    Args:
        activations: [n_samples, d_model] activation vectors.
        d_model: Activation dimension.
        n_features: Number of SAE features (dictionary size).
        l1_coeff: L1 sparsity penalty coefficient.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        val_frac: Fraction for validation.
        normalize_decoder: Re-normalize decoder rows after each step.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        SAETrainResult with trained SAE and metrics.
    """
    activations = jnp.array(activations)
    n_samples = activations.shape[0]

    # Train/val split
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_samples)
    n_val = max(1, int(n_samples * val_frac))
    x_val = activations[perm[:n_val]]
    x_train = activations[perm[n_val:]]
    n_train = x_train.shape[0]

    # Create SAE
    key, subkey = jax.random.split(key)
    sae = SparseAutoencoder(d_model, n_features, key=subkey)

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(sae, eqx.is_array))

    def loss_fn(sae, x):
        x_hat, feature_acts = sae(x)
        recon_loss = jnp.mean((x - x_hat) ** 2)
        l1_loss = jnp.mean(jnp.abs(feature_acts))
        return recon_loss + l1_coeff * l1_loss, (recon_loss, l1_loss, feature_acts)

    @eqx.filter_jit
    def step(sae, opt_state, x):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(sae, x)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(sae, eqx.is_array)
        )
        sae = eqx.apply_updates(sae, updates)
        return sae, new_opt_state, loss, aux

    train_losses = []
    recon_losses = []
    l1_losses = []
    val_losses = []
    l0_sparsities = []

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        train_perm = jax.random.permutation(subkey, n_train)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = train_perm[start:end]
            x_batch = x_train[batch_idx]

            sae, opt_state, batch_loss, (recon, l1, feat_acts) = step(
                sae, opt_state, x_batch
            )

            if normalize_decoder:
                sae = _normalize_decoder(sae)

            epoch_loss += float(batch_loss)
            epoch_recon += float(recon)
            epoch_l1 += float(l1)
            # L0: average number of non-zero features
            epoch_l0 += float(jnp.mean(jnp.sum(feat_acts > 0, axis=-1)))
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)
        recon_losses.append(epoch_recon / n_batches)
        l1_losses.append(epoch_l1 / n_batches)
        l0_sparsities.append(epoch_l0 / n_batches)

        # Validation
        val_loss, (val_recon, _, _) = loss_fn(sae, x_val)
        val_losses.append(float(val_loss))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"loss={train_losses[-1]:.4f} "
                f"recon={recon_losses[-1]:.4f} "
                f"l1={l1_losses[-1]:.4f} "
                f"L0={l0_sparsities[-1]:.1f} "
                f"val={val_losses[-1]:.4f}"
            )

    return SAETrainResult(
        sae=sae,
        train_losses=train_losses,
        recon_losses=recon_losses,
        l1_losses=l1_losses,
        val_losses=val_losses,
        l0_sparsities=l0_sparsities,
    )


# ─── Feature Analysis ────────────────────────────────────────────────────────


def feature_activation_stats(
    sae: SparseAutoencoder,
    activations: jnp.ndarray,
) -> dict:
    """Compute statistics about feature activations.

    Args:
        sae: Trained sparse autoencoder.
        activations: [n_samples, d_model] activation vectors.

    Returns:
        Dict with:
            "mean_acts": [n_features] mean activation per feature.
            "firing_rate": [n_features] fraction of samples where feature fires.
            "max_acts": [n_features] max activation per feature.
            "l0_mean": mean number of active features per sample.
    """
    feat_acts = sae.encode(activations)  # [n_samples, n_features]
    return {
        "mean_acts": np.array(jnp.mean(feat_acts, axis=0)),
        "firing_rate": np.array(jnp.mean(feat_acts > 0, axis=0)),
        "max_acts": np.array(jnp.max(feat_acts, axis=0)),
        "l0_mean": float(jnp.mean(jnp.sum(feat_acts > 0, axis=-1))),
    }


def top_activating_examples(
    sae: SparseAutoencoder,
    feature_idx: int,
    activations: jnp.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the top-k examples that most strongly activate a feature.

    Args:
        sae: Trained sparse autoencoder.
        feature_idx: Which feature to analyze.
        activations: [n_samples, d_model] activation vectors.
        k: Number of top examples.

    Returns:
        (indices, activation_values): indices into activations array and
        the feature's activation value for those examples.
    """
    feat_acts = sae.encode(activations)[:, feature_idx]  # [n_samples]
    k = min(k, len(feat_acts))
    top_idx = np.argsort(np.array(feat_acts))[::-1][:k]
    return top_idx, np.array(feat_acts[top_idx])


def feature_logit_attribution(
    sae: SparseAutoencoder,
    W_U: jnp.ndarray,
    feature_idx: int,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the effect of a feature on output logits.

    Projects the feature direction through the unembedding matrix to see
    which tokens a feature promotes/suppresses.

    Args:
        sae: Trained sparse autoencoder.
        W_U: [d_model, d_vocab] unembedding matrix.
        feature_idx: Which feature to analyze.
        k: Number of top promoted/suppressed tokens.

    Returns:
        (top_promoted_indices, top_suppressed_indices): token indices most
        promoted and suppressed by this feature.
    """
    feature_dir = sae.W_dec[feature_idx]  # [d_model]
    logit_effect = feature_dir @ W_U  # [d_vocab]
    logit_effect_np = np.array(logit_effect)

    promoted = np.argsort(logit_effect_np)[::-1][:k]
    suppressed = np.argsort(logit_effect_np)[:k]
    return promoted, suppressed
