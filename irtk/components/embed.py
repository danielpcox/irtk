"""Token and positional embedding components."""

from typing import Optional

import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState


class Embed(eqx.Module):
    """Token embedding: looks up rows of W_E by token ID.

    W_E shape: [d_vocab, d_model]
    Input: [seq_len] integer token IDs (unbatched)
    Output: [seq_len, d_model]
    """

    W_E: jnp.ndarray
    hook_embed: HookPoint

    def __init__(self, d_vocab: int, d_model: int, *, key=None):
        self.W_E = jnp.zeros((d_vocab, d_model))
        self.hook_embed = HookPoint(name="hook_embed")

    def __call__(
        self, tokens: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        embed = self.W_E[tokens]  # [seq_len, d_model]
        return self.hook_embed(embed, hook_state)


class PosEmbed(eqx.Module):
    """Positional embedding: slices W_pos up to sequence length.

    W_pos shape: [n_ctx, d_model]
    Input: [seq_len] integer positions (or just seq_len inferred from tokens)
    Output: [seq_len, d_model]
    """

    W_pos: jnp.ndarray
    hook_pos_embed: HookPoint

    def __init__(self, n_ctx: int, d_model: int, *, key=None):
        self.W_pos = jnp.zeros((n_ctx, d_model))
        self.hook_pos_embed = HookPoint(name="hook_pos_embed")

    def __call__(
        self, tokens: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        seq_len = tokens.shape[0]
        pos_embed = self.W_pos[:seq_len]  # [seq_len, d_model]
        return self.hook_pos_embed(pos_embed, hook_state)
