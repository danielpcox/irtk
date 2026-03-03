"""MLP components for transformer blocks.

Standard MLP: x -> act(x @ W_in + b_in) @ W_out + b_out
Gated MLP:    x -> (act(x @ W_gate) * (x @ W_in + b_in)) @ W_out + b_out

Operates on [seq_len, d_model] tensors (not vmapped per-token).
"""

from typing import Optional

import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.utilities.activation_fns import ACTIVATION_FN_DICT


class MLP(eqx.Module):
    """Standard MLP with hooks on pre and post activations.

    W_in:  [d_model, d_mlp]
    b_in:  [d_mlp]
    W_out: [d_mlp, d_model]
    b_out: [d_model]
    """

    W_in: jnp.ndarray
    b_in: jnp.ndarray
    W_out: jnp.ndarray
    b_out: jnp.ndarray
    act_fn_name: str = eqx.field(static=True)
    hook_pre: HookPoint
    hook_post: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, *, layer_idx: int):
        d_model = cfg.d_model
        d_mlp = cfg.d_mlp

        self.W_in = jnp.zeros((d_model, d_mlp))
        self.b_in = jnp.zeros(d_mlp)
        self.W_out = jnp.zeros((d_mlp, d_model))
        self.b_out = jnp.zeros(d_model)
        self.act_fn_name = cfg.act_fn

        prefix = f"blocks.{layer_idx}.mlp."
        self.hook_pre = HookPoint(name=f"{prefix}hook_pre")
        self.hook_post = HookPoint(name=f"{prefix}hook_post")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        pre = x @ self.W_in + self.b_in  # [seq_len, d_mlp]
        pre = self.hook_pre(pre, hook_state)
        post = ACTIVATION_FN_DICT[self.act_fn_name](pre)
        post = self.hook_post(post, hook_state)
        return post @ self.W_out + self.b_out


class GatedMLP(eqx.Module):
    """Gated MLP (used by LLaMA, Mistral).

    Applies: (act(x @ W_gate) * (x @ W_in)) @ W_out

    W_gate: [d_model, d_mlp]
    W_in:   [d_model, d_mlp]
    W_out:  [d_mlp, d_model]
    b_out:  [d_model] (some architectures have no biases; we keep it for generality)
    """

    W_gate: jnp.ndarray
    W_in: jnp.ndarray
    W_out: jnp.ndarray
    b_out: jnp.ndarray
    act_fn_name: str = eqx.field(static=True)
    hook_pre: HookPoint
    hook_post: HookPoint
    hook_pre_linear: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, *, layer_idx: int):
        d_model = cfg.d_model
        d_mlp = cfg.d_mlp

        self.W_gate = jnp.zeros((d_model, d_mlp))
        self.W_in = jnp.zeros((d_model, d_mlp))
        self.W_out = jnp.zeros((d_mlp, d_model))
        self.b_out = jnp.zeros(d_model)
        self.act_fn_name = cfg.act_fn

        prefix = f"blocks.{layer_idx}.mlp."
        self.hook_pre = HookPoint(name=f"{prefix}hook_pre")
        self.hook_post = HookPoint(name=f"{prefix}hook_post")
        self.hook_pre_linear = HookPoint(name=f"{prefix}hook_pre_linear")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        pre = x @ self.W_gate  # [seq_len, d_mlp]
        pre = self.hook_pre(pre, hook_state)
        gate = ACTIVATION_FN_DICT[self.act_fn_name](pre)
        pre_linear = x @ self.W_in  # [seq_len, d_mlp]
        pre_linear = self.hook_pre_linear(pre_linear, hook_state)
        post = gate * pre_linear
        post = self.hook_post(post, hook_state)
        return post @ self.W_out + self.b_out
