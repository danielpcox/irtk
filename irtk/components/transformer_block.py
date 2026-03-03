"""Transformer block: attention + MLP with residual connections and layer norms.

Supports:
- Sequential (standard): LN -> Attn -> Add -> LN -> MLP -> Add
- Parallel (GPT-J style): LN -> (Attn + MLP) -> Add
"""

from typing import Optional

import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.components.attention import Attention
from irtk.components.mlps import MLP, GatedMLP
from irtk.components.layer_norm import get_layer_norm


class TransformerBlock(eqx.Module):
    """A single transformer block with full hook coverage."""

    attn: Attention
    mlp: eqx.Module  # MLP or GatedMLP
    ln1: Optional[eqx.Module]  # pre-attention layer norm
    ln2: Optional[eqx.Module]  # pre-MLP layer norm (None for parallel)
    parallel_attn_mlp: bool = eqx.field(static=True)

    hook_resid_pre: HookPoint
    hook_resid_mid: HookPoint   # only used in sequential mode
    hook_resid_post: HookPoint
    hook_attn_out: HookPoint
    hook_mlp_out: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, *, layer_idx: int):
        self.parallel_attn_mlp = cfg.parallel_attn_mlp

        self.attn = Attention(cfg, layer_idx=layer_idx)

        if cfg.gated_mlp:
            self.mlp = GatedMLP(cfg, layer_idx=layer_idx)
        else:
            self.mlp = MLP(cfg, layer_idx=layer_idx)

        prefix = f"blocks.{layer_idx}."
        ln_prefix1 = f"{prefix}ln1."
        ln_prefix2 = f"{prefix}ln2."

        self.ln1 = get_layer_norm(
            cfg.normalization_type, cfg.d_model, cfg.eps, name_prefix=ln_prefix1
        )
        if not cfg.parallel_attn_mlp:
            self.ln2 = get_layer_norm(
                cfg.normalization_type, cfg.d_model, cfg.eps, name_prefix=ln_prefix2
            )
        else:
            self.ln2 = None

        self.hook_resid_pre = HookPoint(name=f"{prefix}hook_resid_pre")
        self.hook_resid_mid = HookPoint(name=f"{prefix}hook_resid_mid")
        self.hook_resid_post = HookPoint(name=f"{prefix}hook_resid_post")
        self.hook_attn_out = HookPoint(name=f"{prefix}hook_attn_out")
        self.hook_mlp_out = HookPoint(name=f"{prefix}hook_mlp_out")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        """Forward pass.

        x: [seq_len, d_model]
        Returns: [seq_len, d_model]
        """
        resid = self.hook_resid_pre(x, hook_state)

        if self.parallel_attn_mlp:
            # GPT-J style: LN -> (Attn + MLP) -> Add
            normed = self.ln1(resid, hook_state) if self.ln1 is not None else resid
            attn_out = self.attn(normed, hook_state)
            attn_out = self.hook_attn_out(attn_out, hook_state)
            mlp_out = self.mlp(normed, hook_state)
            mlp_out = self.hook_mlp_out(mlp_out, hook_state)
            resid = resid + attn_out + mlp_out
        else:
            # Standard: LN1 -> Attn -> Add -> LN2 -> MLP -> Add
            normed1 = self.ln1(resid, hook_state) if self.ln1 is not None else resid
            attn_out = self.attn(normed1, hook_state)
            attn_out = self.hook_attn_out(attn_out, hook_state)
            resid = resid + attn_out
            resid = self.hook_resid_mid(resid, hook_state)

            normed2 = self.ln2(resid, hook_state) if self.ln2 is not None else resid
            mlp_out = self.mlp(normed2, hook_state)
            mlp_out = self.hook_mlp_out(mlp_out, hook_state)
            resid = resid + mlp_out

        resid = self.hook_resid_post(resid, hook_state)
        return resid
