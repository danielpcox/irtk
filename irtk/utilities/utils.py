"""Utility functions for TransformerLens in JAX."""

import jax
import jax.numpy as jnp


def get_act_name(name: str, layer: int | None = None) -> str:
    """Get the full hook name for an activation.

    Args:
        name: Short activation name (e.g., "q", "k", "resid_pre", "mlp_out").
        layer: Layer index. None for non-layer activations (embed, pos_embed, ln_final).

    Returns:
        Full hook name like "blocks.3.hook_resid_pre" or "hook_embed".
    """
    if layer is None:
        return f"hook_{name}"
    return f"blocks.{layer}.hook_{name}"


def lm_cross_entropy_loss(
    logits: jnp.ndarray, tokens: jnp.ndarray, per_token: bool = False
) -> jnp.ndarray:
    """Cross-entropy loss for language modeling.

    Args:
        logits: [seq_len, d_vocab] predicted logits (or [batch, seq_len, d_vocab]).
        tokens: [seq_len] or [batch, seq_len] target token IDs.
        per_token: If True, return per-token loss instead of mean.

    Returns:
        Scalar loss (or per-token losses if per_token=True).
    """
    # Shift: predict next token from each position
    # logits[:-1] predicts tokens[1:]
    log_probs = jax.nn.log_softmax(logits[..., :-1, :], axis=-1)
    target_tokens = tokens[..., 1:]
    # Gather log probs for target tokens
    loss = -jnp.take_along_axis(
        log_probs, target_tokens[..., None], axis=-1
    ).squeeze(-1)
    if per_token:
        return loss
    return loss.mean()


def lm_accuracy(logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
    """Compute next-token prediction accuracy.

    Args:
        logits: [seq_len, d_vocab] or [batch, seq_len, d_vocab].
        tokens: [seq_len] or [batch, seq_len] target token IDs.

    Returns:
        Scalar accuracy.
    """
    predictions = jnp.argmax(logits[..., :-1, :], axis=-1)
    targets = tokens[..., 1:]
    return (predictions == targets).mean()
