"""Unembedding component: projects from residual stream to vocabulary logits."""

from typing import Optional

import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState


class Unembed(eqx.Module):
    """Unembedding: x @ W_U + b_U

    W_U shape: [d_model, d_vocab_out]
    b_U shape: [d_vocab_out]
    Input: [seq_len, d_model]
    Output: [seq_len, d_vocab_out]
    """

    W_U: jnp.ndarray
    b_U: jnp.ndarray

    def __init__(self, d_model: int, d_vocab_out: int, *, key=None):
        self.W_U = jnp.zeros((d_model, d_vocab_out))
        self.b_U = jnp.zeros(d_vocab_out)

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        return x @ self.W_U + self.b_U
