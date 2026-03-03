"""Transcoders: sparse MLP replacements for interpretable feature circuits.

Transcoders learn a sparse, overcomplete mapping from MLP input to MLP output,
enabling weights-based circuit analysis through MLP layers (unlike SAEs which
only reconstruct residual stream activations).

References:
    - Dunefsky et al. (2024) "Transcoders Find Interpretable LLM Feature Circuits" (NeurIPS 2024)
    - Anthropic (2025) Circuit Tracing / Attribution Graphs
"""

from typing import Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from irtk.hooked_transformer import HookedTransformer


class Transcoder(eqx.Module):
    """Sparse transcoder: maps MLP input -> MLP output via a wide sparse hidden layer.

    Architecture:
        encode: (x - b_dec) @ W_in + b_enc -> ReLU -> feature_acts
        decode: feature_acts @ W_out + b_dec -> y_hat (approximates MLP output)

    Unlike SAEs, this maps input -> output (not input -> reconstructed input).
    """

    W_in: jnp.ndarray    # [d_model, n_features]
    b_enc: jnp.ndarray   # [n_features]
    W_out: jnp.ndarray   # [n_features, d_model]
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
        self.W_in = jax.random.normal(k1, (d_model, n_features)) * (1.0 / jnp.sqrt(d_model))
        self.W_out = jax.random.normal(k2, (n_features, d_model)) * (1.0 / jnp.sqrt(n_features))
        self.b_enc = jnp.zeros(n_features)
        self.b_dec = jnp.zeros(d_model)

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode MLP input to sparse feature activations.

        Args:
            x: [..., d_model] MLP input activations.

        Returns:
            [..., n_features] sparse feature activations (after ReLU).
        """
        x_centered = x - self.b_dec
        return jax.nn.relu(x_centered @ self.W_in + self.b_enc)

    def decode(self, feature_acts: jnp.ndarray) -> jnp.ndarray:
        """Decode feature activations to MLP output space.

        Args:
            feature_acts: [..., n_features] sparse feature activations.

        Returns:
            [..., d_model] predicted MLP output.
        """
        return feature_acts @ self.W_out + self.b_dec

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Full forward pass: encode then decode.

        Args:
            x: [..., d_model] MLP input activations.

        Returns:
            (y_hat, feature_acts): predicted MLP output and sparse features.
        """
        feature_acts = self.encode(x)
        y_hat = self.decode(feature_acts)
        return y_hat, feature_acts


@dataclass
class TranscoderTrainResult:
    """Results from training a transcoder."""

    transcoder: Transcoder
    train_losses: list[float]
    recon_losses: list[float]
    l1_losses: list[float]
    val_losses: list[float]
    l0_sparsities: list[float]


def train_transcoder(
    model: HookedTransformer,
    layer: int,
    token_sequences: list,
    n_features: int = 512,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    epochs: int = 10,
    batch_size: int = 256,
    val_frac: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> TranscoderTrainResult:
    """Train a transcoder to approximate an MLP layer with sparse features.

    Collects (mlp_input, mlp_output) pairs by running token_sequences through
    the model, then trains the transcoder to minimize
    ||mlp_output - transcoder(mlp_input)||^2 + l1 * ||feature_acts||_1.

    Args:
        model: HookedTransformer.
        layer: Which MLP layer to approximate.
        token_sequences: List of token arrays.
        n_features: Number of transcoder features.
        l1_coeff: L1 sparsity coefficient.
        lr: Learning rate.
        epochs: Training epochs.
        batch_size: Batch size.
        val_frac: Validation fraction.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        TranscoderTrainResult with trained transcoder and metrics.
    """
    d_model = model.cfg.d_model

    # Collect (input, output) pairs
    input_hook = f"blocks.{layer}.hook_resid_mid"  # input to MLP
    output_hook = f"blocks.{layer}.hook_mlp_out"

    all_inputs = []
    all_outputs = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if input_hook in cache.cache_dict and output_hook in cache.cache_dict:
            all_inputs.append(np.array(cache.cache_dict[input_hook]))
            all_outputs.append(np.array(cache.cache_dict[output_hook]))

    if not all_inputs:
        # Fallback: use resid_pre as input
        input_hook = f"blocks.{layer}.hook_resid_pre"
        for tokens in token_sequences:
            tokens = jnp.array(tokens)
            _, cache = model.run_with_cache(tokens)
            if input_hook in cache.cache_dict and output_hook in cache.cache_dict:
                all_inputs.append(np.array(cache.cache_dict[input_hook]))
                all_outputs.append(np.array(cache.cache_dict[output_hook]))

    x_in = jnp.concatenate([jnp.array(a.reshape(-1, d_model)) for a in all_inputs], axis=0)
    x_out = jnp.concatenate([jnp.array(a.reshape(-1, d_model)) for a in all_outputs], axis=0)
    n_samples = x_in.shape[0]

    # Train/val split
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_samples)
    n_val = max(1, int(n_samples * val_frac))
    in_val, out_val = x_in[perm[:n_val]], x_out[perm[:n_val]]
    in_train, out_train = x_in[perm[n_val:]], x_out[perm[n_val:]]
    n_train = in_train.shape[0]

    # Create transcoder
    key, subkey = jax.random.split(key)
    tc = Transcoder(d_model, n_features, key=subkey)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(tc, eqx.is_array))

    def loss_fn(tc, x_in, x_out):
        y_hat, feat_acts = tc(x_in)
        recon_loss = jnp.mean((x_out - y_hat) ** 2)
        l1_loss = jnp.mean(jnp.abs(feat_acts))
        return recon_loss + l1_coeff * l1_loss, (recon_loss, l1_loss, feat_acts)

    @eqx.filter_jit
    def step(tc, opt_state, x_in, x_out):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(tc, x_in, x_out)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(tc, eqx.is_array)
        )
        tc = eqx.apply_updates(tc, updates)
        return tc, new_opt_state, loss, aux

    train_losses, recon_losses, l1_losses, val_losses, l0_sparsities = [], [], [], [], []

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        train_perm = jax.random.permutation(subkey, n_train)

        epoch_loss = epoch_recon = epoch_l1 = epoch_l0 = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = train_perm[start:end]
            tc, opt_state, batch_loss, (recon, l1, feat_acts) = step(
                tc, opt_state, in_train[idx], out_train[idx]
            )
            epoch_loss += float(batch_loss)
            epoch_recon += float(recon)
            epoch_l1 += float(l1)
            epoch_l0 += float(jnp.mean(jnp.sum(feat_acts > 0, axis=-1)))
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)
        recon_losses.append(epoch_recon / n_batches)
        l1_losses.append(epoch_l1 / n_batches)
        l0_sparsities.append(epoch_l0 / n_batches)

        val_loss, _ = loss_fn(tc, in_val, out_val)
        val_losses.append(float(val_loss))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d}: loss={train_losses[-1]:.4f} "
                  f"recon={recon_losses[-1]:.4f} l1={l1_losses[-1]:.4f} "
                  f"L0={l0_sparsities[-1]:.1f} val={val_losses[-1]:.4f}")

    return TranscoderTrainResult(
        transcoder=tc,
        train_losses=train_losses,
        recon_losses=recon_losses,
        l1_losses=l1_losses,
        val_losses=val_losses,
        l0_sparsities=l0_sparsities,
    )


def transcoder_feature_circuit(
    transcoder_a: Transcoder,
    transcoder_b: Transcoder,
) -> dict:
    """Build a weight-based feature-to-feature connection matrix between two transcoders.

    Computes connection[i,j] = transcoder_a.W_out[i] @ transcoder_b.W_in[:, j],
    showing how features in layer A compose with features in layer B via
    the residual stream, without running any forward passes.

    Args:
        transcoder_a: Transcoder at an earlier layer.
        transcoder_b: Transcoder at a later layer.

    Returns:
        Dict with:
        - "connection_matrix": [n_features_a, n_features_b] direct connection strengths
        - "top_connections": list of (feat_a, feat_b, strength) top connections
        - "mean_connection": mean absolute connection strength
    """
    # connection[i,j] = W_out_a[i] @ W_in_b[:, j]
    W_out_a = np.array(transcoder_a.W_out)  # [n_a, d_model]
    W_in_b = np.array(transcoder_b.W_in)    # [d_model, n_b]

    connection = W_out_a @ W_in_b  # [n_a, n_b]

    # Find top connections
    flat_idx = np.argsort(np.abs(connection).ravel())[::-1][:20]
    n_b = connection.shape[1]
    top_connections = []
    for idx in flat_idx:
        i, j = divmod(int(idx), n_b)
        top_connections.append((i, j, float(connection[i, j])))

    return {
        "connection_matrix": connection,
        "top_connections": top_connections,
        "mean_connection": float(np.mean(np.abs(connection))),
    }


def top_activating_for_transcoder_feature(
    transcoder: Transcoder,
    model: HookedTransformer,
    layer: int,
    feature_idx: int,
    token_sequences: list,
    k: int = 20,
) -> list[dict]:
    """Find top-k token contexts that maximally activate a transcoder feature.

    Args:
        transcoder: Trained Transcoder.
        model: HookedTransformer.
        layer: Which layer the transcoder is for.
        feature_idx: Feature to analyze.
        token_sequences: List of token arrays to scan.
        k: Number of top examples.

    Returns:
        List of dicts sorted by activation (descending), each with:
        - "prompt_idx": which token sequence
        - "position": sequence position
        - "activation": feature activation value
        - "tokens": the full token sequence
    """
    results = []

    # Try resid_mid first (MLP input), fall back to resid_pre
    input_hook = f"blocks.{layer}.hook_resid_mid"

    for prompt_idx, tokens in enumerate(token_sequences):
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        hook = input_hook
        if hook not in cache.cache_dict:
            hook = f"blocks.{layer}.hook_resid_pre"
        if hook not in cache.cache_dict:
            continue

        acts = cache.cache_dict[hook]  # [seq_len, d_model]
        feat_acts = np.array(transcoder.encode(acts))  # [seq_len, n_features]
        feat_col = feat_acts[:, feature_idx]

        for pos in range(len(feat_col)):
            results.append({
                "prompt_idx": prompt_idx,
                "position": int(pos),
                "activation": float(feat_col[pos]),
                "tokens": np.array(tokens),
            })

    results.sort(key=lambda x: x["activation"], reverse=True)
    return results[:k]


def mlp_feature_logit_attribution(
    transcoder: Transcoder,
    model: HookedTransformer,
    feature_idx: int,
    k: int = 20,
) -> dict:
    """Project a transcoder feature through unembedding to get logit effect.

    Computes transcoder.W_out[feature_idx] @ W_U to see which output tokens
    a feature promotes or suppresses.

    Args:
        transcoder: Trained Transcoder.
        model: HookedTransformer.
        feature_idx: Feature to analyze.
        k: Number of top promoted/suppressed tokens.

    Returns:
        Dict with:
        - "top_promoted": [(token_id, logit_effect), ...] tokens promoted
        - "top_suppressed": [(token_id, logit_effect), ...] tokens suppressed
        - "logit_effects": [d_vocab] full logit effect vector
    """
    feat_dir = np.array(transcoder.W_out[feature_idx])  # [d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    effects = feat_dir @ W_U  # [d_vocab]

    top_promoted_idx = np.argsort(effects)[::-1][:k]
    top_suppressed_idx = np.argsort(effects)[:k]

    return {
        "top_promoted": [(int(i), float(effects[i])) for i in top_promoted_idx],
        "top_suppressed": [(int(i), float(effects[i])) for i in top_suppressed_idx],
        "logit_effects": effects,
    }
