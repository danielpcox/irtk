"""Multi-head attention with support for GQA and rotary embeddings.

Param shapes follow TransformerLens convention for per-head analysis:
  W_Q, W_K, W_V: [n_heads, d_model, d_head]
  b_Q, b_K, b_V: [n_heads, d_head]
  W_O: [n_heads, d_head, d_model]
  b_O: [d_model]

Input: [seq_len, d_model] (unbatched)
Output: [seq_len, d_model]
"""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState
from irtk.hooked_transformer_config import HookedTransformerConfig


def _apply_rotary_embed(
    x: jnp.ndarray, seq_len: int, rotary_dim: int, rotary_base: float
) -> jnp.ndarray:
    """Apply rotary positional embeddings to x.

    x: [seq_len, n_heads, d_head]
    Returns: same shape with rotary applied to first rotary_dim dims.
    """
    positions = jnp.arange(seq_len)
    dim_pairs = rotary_dim // 2
    freq_seq = jnp.arange(dim_pairs, dtype=jnp.float32)
    freqs = 1.0 / (rotary_base ** (freq_seq / dim_pairs))
    # [seq_len, dim_pairs]
    angles = jnp.outer(positions, freqs)
    cos = jnp.cos(angles)  # [seq_len, dim_pairs]
    sin = jnp.sin(angles)

    # Split x into rotary and passthrough parts
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Split rotary part into pairs
    x1 = x_rot[..., ::2]   # [seq_len, n_heads, dim_pairs]
    x2 = x_rot[..., 1::2]  # [seq_len, n_heads, dim_pairs]

    # Broadcast cos/sin: [seq_len, 1, dim_pairs]
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    rotated = jnp.stack([out1, out2], axis=-1).reshape(x_rot.shape)
    return jnp.concatenate([rotated, x_pass], axis=-1)


class Attention(eqx.Module):
    """Multi-head attention with hooks at every intermediate activation.

    Supports:
    - Standard multi-head attention
    - Grouped query attention (GQA) via n_key_value_heads
    - Rotary positional embeddings
    - Causal masking
    """

    W_Q: jnp.ndarray   # [n_heads, d_model, d_head]
    W_K: jnp.ndarray   # [n_kv_heads, d_model, d_head]
    W_V: jnp.ndarray   # [n_kv_heads, d_model, d_head]
    W_O: jnp.ndarray   # [n_heads, d_head, d_model]
    b_Q: jnp.ndarray   # [n_heads, d_head]
    b_K: jnp.ndarray   # [n_kv_heads, d_head]
    b_V: jnp.ndarray   # [n_kv_heads, d_head]
    b_O: jnp.ndarray   # [d_model]

    # Config (static)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    attn_scale: float = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    use_rotary: bool = eqx.field(static=True)
    rotary_dim: int = eqx.field(static=True)
    rotary_base: float = eqx.field(static=True)

    # Hook points
    hook_q: HookPoint
    hook_k: HookPoint
    hook_v: HookPoint
    hook_z: HookPoint
    hook_attn_scores: HookPoint
    hook_pattern: HookPoint
    hook_result: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, *, layer_idx: int):
        n_heads = cfg.n_heads
        n_kv = cfg.n_key_value_heads
        d_model = cfg.d_model
        d_head = cfg.d_head

        self.n_heads = n_heads
        self.n_kv_heads = n_kv
        self.d_head = d_head
        self.d_model = d_model
        self.attn_scale = cfg.attn_scale
        self.causal = cfg.attention_dir == "causal"
        self.use_rotary = cfg.positional_embedding_type == "rotary"
        self.rotary_dim = cfg.rotary_dim if cfg.rotary_dim is not None else 0
        self.rotary_base = cfg.rotary_base

        # Initialize weight matrices (zeros; will be overwritten by pretrained weights)
        self.W_Q = jnp.zeros((n_heads, d_model, d_head))
        self.W_K = jnp.zeros((n_kv, d_model, d_head))
        self.W_V = jnp.zeros((n_kv, d_model, d_head))
        self.W_O = jnp.zeros((n_heads, d_head, d_model))
        self.b_Q = jnp.zeros((n_heads, d_head))
        self.b_K = jnp.zeros((n_kv, d_head))
        self.b_V = jnp.zeros((n_kv, d_head))
        self.b_O = jnp.zeros(d_model)

        # Hook points
        prefix = f"blocks.{layer_idx}.attn."
        self.hook_q = HookPoint(name=f"{prefix}hook_q")
        self.hook_k = HookPoint(name=f"{prefix}hook_k")
        self.hook_v = HookPoint(name=f"{prefix}hook_v")
        self.hook_z = HookPoint(name=f"{prefix}hook_z")
        self.hook_attn_scores = HookPoint(name=f"{prefix}hook_attn_scores")
        self.hook_pattern = HookPoint(name=f"{prefix}hook_pattern")
        self.hook_result = HookPoint(name=f"{prefix}hook_result")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        """Forward pass.

        x: [seq_len, d_model]
        Returns: [seq_len, d_model]
        """
        seq_len = x.shape[0]

        # Compute Q, K, V
        # x:[s,m] W_Q:[n,m,h] -> q:[s,n,h]  (m=d_model, h=d_head, n=n_heads)
        q = jnp.einsum("sm,nmh->snh", x, self.W_Q) + self.b_Q  # [seq, n_heads, d_head]
        k = jnp.einsum("sm,nmh->snh", x, self.W_K) + self.b_K  # [seq, n_kv, d_head]
        v = jnp.einsum("sm,nmh->snh", x, self.W_V) + self.b_V  # [seq, n_kv, d_head]

        # Apply rotary embeddings
        if self.use_rotary:
            q = _apply_rotary_embed(q, seq_len, self.rotary_dim, self.rotary_base)
            k = _apply_rotary_embed(k, seq_len, self.rotary_dim, self.rotary_base)

        q = self.hook_q(q, hook_state)
        k = self.hook_k(k, hook_state)
        v = self.hook_v(v, hook_state)

        # Expand K/V for GQA: repeat n_kv heads to match n_heads
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, repeats, axis=1)  # [seq, n_heads, d_head]
            v = jnp.repeat(v, repeats, axis=1)

        # Attention scores: [n_heads, seq_q, seq_k]
        # q:[q,n,h] k:[k,n,h] -> [n,q,k]
        attn_scores = jnp.einsum("qnh,knh->nqk", q, k) / self.attn_scale
        attn_scores = self.hook_attn_scores(attn_scores, hook_state)

        # Causal masking
        if self.causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attn_scores = jnp.where(mask[None, :, :], attn_scores, jnp.finfo(attn_scores.dtype).min)

        # Softmax -> attention pattern
        pattern = jax.nn.softmax(attn_scores, axis=-1)
        pattern = self.hook_pattern(pattern, hook_state)

        # Weighted sum of values: [seq, n_heads, d_head]
        # pattern:[n,q,k] v:[k,n,h] -> z:[q,n,h]
        z = jnp.einsum("nqk,knh->qnh", pattern, v)
        z = self.hook_z(z, hook_state)

        # Output projection: z:[s,n,h] W_O:[n,h,m] -> [s,m]
        result = jnp.einsum("snh,nhm->sm", z, self.W_O) + self.b_O
        result = self.hook_result(result, hook_state)
        return result
