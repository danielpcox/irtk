"""Logit lens and tuned lens for inspecting intermediate representations.

The logit lens (nostalgebraist, 2020) projects intermediate residual stream
states through the unembedding matrix to see what the model "thinks" at each
layer. The tuned lens (Belrose et al., 2023) learns an affine probe per layer
to improve these projections.
"""

from typing import Optional, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.activation_cache import ActivationCache


# ─── Logit Lens ──────────────────────────────────────────────────────────────


def logit_lens(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    apply_ln: bool = True,
    return_probs: bool = False,
) -> np.ndarray:
    """Apply the logit lens to get per-layer token predictions.

    Projects the residual stream at each layer through the unembedding to
    get a distribution over vocabulary tokens at each position and layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] integer token IDs.
        apply_ln: If True, apply final layer norm before unembedding.
        return_probs: If True, return probabilities instead of logits.

    Returns:
        [n_layers+1, seq_len, d_vocab] array of logits (or probs).
        Index 0 = after embedding, index i+1 = after layer i.
    """
    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()  # [n_components, seq_len, d_model]

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    if apply_ln and model.ln_final is not None:
        # Apply final layer norm to each layer's residual
        ln = model.ln_final
        normed = jnp.stack([ln(resid_stack[i]) for i in range(resid_stack.shape[0])])
        logits = jnp.einsum("csd,dv->csv", normed, W_U) + b_U
    else:
        logits = jnp.einsum("csd,dv->csv", resid_stack, W_U) + b_U

    if return_probs:
        return np.array(jax.nn.softmax(logits, axis=-1))
    return np.array(logits)


def logit_lens_top_k(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    k: int = 5,
    apply_ln: bool = True,
) -> list[list[list[tuple[int, float]]]]:
    """Get top-k token predictions at each layer and position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        k: Number of top predictions.
        apply_ln: Apply final layer norm before unembedding.

    Returns:
        Nested list: [layer][position] -> list of (token_id, probability) tuples.
    """
    probs = logit_lens(model, tokens, apply_ln=apply_ln, return_probs=True)
    n_components, seq_len, _ = probs.shape

    results = []
    for layer in range(n_components):
        layer_results = []
        for pos in range(seq_len):
            top_indices = np.argsort(probs[layer, pos])[::-1][:k]
            top_probs = [(int(idx), float(probs[layer, pos, idx])) for idx in top_indices]
            layer_results.append(top_probs)
        results.append(layer_results)
    return results


def logit_lens_correct_prob(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    apply_ln: bool = True,
) -> np.ndarray:
    """Get probability assigned to the correct next token at each layer/position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        apply_ln: Apply final layer norm.

    Returns:
        [n_layers+1, seq_len-1] array of probabilities for the correct next token.
        Layer i, position j gives P(tokens[j+1]) using the residual after layer i
        at position j.
    """
    probs = logit_lens(model, tokens, apply_ln=apply_ln, return_probs=True)
    # probs: [n_components, seq_len, d_vocab]
    # For position j, the "correct" next token is tokens[j+1]
    target_tokens = np.array(tokens[1:])  # [seq_len - 1]
    # Gather: for each component and position, get prob of next token
    correct_probs = probs[:, :-1, :]  # [n_components, seq_len-1, d_vocab]
    result = np.array([
        [float(correct_probs[c, p, target_tokens[p]]) for p in range(len(target_tokens))]
        for c in range(correct_probs.shape[0])
    ])
    return result


def logit_lens_kl_divergence(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    apply_ln: bool = True,
) -> np.ndarray:
    """Compute KL divergence between each layer's prediction and the final layer.

    A measure of how much each layer's "belief" differs from the model's output.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        apply_ln: Apply final layer norm.

    Returns:
        [n_layers, seq_len] array of KL divergences from final-layer distribution.
        (Excludes the final layer itself since KL(final || final) = 0.)
    """
    logits = logit_lens(model, tokens, apply_ln=apply_ln, return_probs=False)
    # logits: [n_components, seq_len, d_vocab]
    log_probs = jax.nn.log_softmax(jnp.array(logits), axis=-1)
    final_log_probs = log_probs[-1]  # [seq_len, d_vocab]
    final_probs = jnp.exp(final_log_probs)

    kl_divs = []
    for i in range(log_probs.shape[0] - 1):
        # KL(final || layer_i) = sum_v final_probs[v] * (final_log_probs[v] - layer_i_log_probs[v])
        kl = jnp.sum(final_probs * (final_log_probs - log_probs[i]), axis=-1)
        kl_divs.append(np.array(kl))

    return np.stack(kl_divs, axis=0)


# ─── Tuned Lens ──────────────────────────────────────────────────────────────


class TunedLensProbe(eqx.Module):
    """A single tuned lens probe: an affine transformation for one layer.

    Initialized as identity + zero bias, so before training it behaves
    like the standard logit lens.
    """

    weight: jnp.ndarray  # [d_model, d_model]
    bias: jnp.ndarray    # [d_model]

    def __init__(self, d_model: int, *, key: jax.random.PRNGKey):
        self.weight = jnp.eye(d_model)
        self.bias = jnp.zeros(d_model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transform residual stream before unembedding.

        Args:
            x: [..., d_model] residual stream activations.

        Returns:
            [..., d_model] transformed activations.
        """
        return x @ self.weight + self.bias


@dataclass
class TunedLens:
    """A collection of tuned lens probes, one per layer.

    After training, use apply() to get improved logit lens predictions.
    """

    probes: list[TunedLensProbe]
    model: HookedTransformer

    def apply(
        self,
        tokens: jnp.ndarray,
        apply_ln: bool = True,
        return_probs: bool = False,
    ) -> np.ndarray:
        """Apply the tuned lens to get per-layer predictions.

        Args:
            tokens: [seq_len] token IDs.
            apply_ln: Apply final layer norm after probe.
            return_probs: Return probabilities instead of logits.

        Returns:
            [n_layers+1, seq_len, d_vocab] logits or probs.
            Index 0 = embedding (no probe), index i+1 = after layer i.
        """
        _, cache = self.model.run_with_cache(tokens)
        resid_stack = cache.accumulated_resid()  # [n_components, seq_len, d_model]

        W_U = self.model.unembed.W_U
        b_U = self.model.unembed.b_U
        ln = self.model.ln_final

        all_logits = []
        for i in range(resid_stack.shape[0]):
            resid = resid_stack[i]
            # Apply tuned lens probe (skip for last component = final layer output)
            if i < len(self.probes):
                resid = self.probes[i](resid)
            if apply_ln and ln is not None:
                resid = ln(resid)
            logits = resid @ W_U + b_U
            all_logits.append(logits)

        result = jnp.stack(all_logits, axis=0)
        if return_probs:
            return np.array(jax.nn.softmax(result, axis=-1))
        return np.array(result)


@dataclass
class TunedLensTrainResult:
    """Results from training a tuned lens."""

    tuned_lens: TunedLens
    train_losses: list[list[float]]  # [n_probes][epochs]
    val_losses: list[list[float]]


def train_tuned_lens(
    model: HookedTransformer,
    train_tokens: jnp.ndarray,
    val_tokens: Optional[jnp.ndarray] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    verbose: bool = True,
) -> TunedLensTrainResult:
    """Train tuned lens probes for a model.

    Each probe is trained independently to minimize KL divergence between
    its output distribution and the model's final output distribution.

    Args:
        model: HookedTransformer.
        train_tokens: [n_train, seq_len] or [seq_len] training token sequences.
        val_tokens: Optional validation tokens.
        epochs: Number of training epochs per probe.
        lr: Learning rate.
        weight_decay: L2 regularization.
        verbose: Print progress.

    Returns:
        TunedLensTrainResult with trained probes and loss history.
    """
    # Handle single-sequence input
    if train_tokens.ndim == 1:
        train_tokens = train_tokens[None, :]

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    ln = model.ln_final

    # Collect residual streams from training data
    all_resids = []  # list of [n_components, seq_len, d_model]
    all_final_log_probs = []  # list of [seq_len, d_vocab]

    for i in range(train_tokens.shape[0]):
        _, cache = model.run_with_cache(train_tokens[i])
        resid = cache.accumulated_resid()
        all_resids.append(resid)

        # Final layer prediction (target)
        final_resid = resid[-1]
        if ln is not None:
            final_resid = ln(final_resid)
        final_logits = final_resid @ W_U + b_U
        all_final_log_probs.append(jax.nn.log_softmax(final_logits, axis=-1))

    # Stack: [n_train, n_components, seq_len, d_model]
    all_resids = jnp.stack(all_resids, axis=0)
    all_final_log_probs = jnp.stack(all_final_log_probs, axis=0)

    # Validation data
    val_resids = None
    val_final_log_probs = None
    if val_tokens is not None:
        if val_tokens.ndim == 1:
            val_tokens = val_tokens[None, :]
        vr = []
        vf = []
        for i in range(val_tokens.shape[0]):
            _, cache = model.run_with_cache(val_tokens[i])
            resid = cache.accumulated_resid()
            vr.append(resid)
            final_resid = resid[-1]
            if ln is not None:
                final_resid = ln(final_resid)
            final_logits = final_resid @ W_U + b_U
            vf.append(jax.nn.log_softmax(final_logits, axis=-1))
        val_resids = jnp.stack(vr, axis=0)
        val_final_log_probs = jnp.stack(vf, axis=0)

    # Train one probe per layer (excluding the final layer output)
    n_probes = all_resids.shape[1] - 1  # embed + each layer except last
    key = jax.random.PRNGKey(0)
    probes = []
    all_train_losses = []
    all_val_losses = []

    for probe_idx in range(n_probes):
        key, subkey = jax.random.split(key)
        probe = TunedLensProbe(d_model, key=subkey)
        optimizer = optax.adamw(lr, weight_decay=weight_decay)
        opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

        # Training data for this probe: residual at this layer for all sequences
        # [n_train, seq_len, d_model]
        layer_resids = all_resids[:, probe_idx]

        def loss_fn(probe, resids, target_log_probs):
            # resids: [n_train, seq_len, d_model]
            # target_log_probs: [n_train, seq_len, d_vocab]
            transformed = jax.vmap(probe)(resids)  # [n_train, seq_len, d_model]
            if ln is not None:
                transformed = jax.vmap(ln)(transformed)
            logits = jnp.einsum("bsd,dv->bsv", transformed, W_U) + b_U
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            # KL(target || predicted) = sum target_probs * (target_log_probs - predicted_log_probs)
            target_probs = jnp.exp(target_log_probs)
            kl = jnp.sum(target_probs * (target_log_probs - log_probs), axis=-1)
            return jnp.mean(kl)

        @eqx.filter_jit
        def step(probe, opt_state, resids, target_log_probs):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(probe, resids, target_log_probs)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(probe, eqx.is_array)
            )
            probe = eqx.apply_updates(probe, updates)
            return probe, new_opt_state, loss

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            probe, opt_state, train_loss = step(
                probe, opt_state, layer_resids, all_final_log_probs
            )
            train_losses.append(float(train_loss))

            if val_resids is not None:
                val_loss = float(loss_fn(probe, val_resids[:, probe_idx], val_final_log_probs))
                val_losses.append(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"  Probe {probe_idx} epoch {epoch:3d}: train_kl={train_losses[-1]:.4f}"
                if val_losses:
                    msg += f" val_kl={val_losses[-1]:.4f}"
                print(msg)

        probes.append(probe)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    tuned_lens = TunedLens(probes=probes, model=model)
    return TunedLensTrainResult(
        tuned_lens=tuned_lens,
        train_losses=all_train_losses,
        val_losses=all_val_losses,
    )
