"""Layer normalization variants.

All operate on [seq_len, d_model] tensors (NOT vmapped per-token),
because hooks must capture full-sequence activations.
Normalization uses axis=-1 for per-token mean/var.
"""

from typing import Optional

import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState


class LayerNorm(eqx.Module):
    """Standard Layer Normalization with learned affine parameters.

    Normalizes, then applies w * x + b. Exposes hook_scale and hook_normalized.
    """

    w: jnp.ndarray  # [d_model]
    b: jnp.ndarray  # [d_model]
    eps: float = eqx.field(static=True)
    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, d_model: int, eps: float = 1e-5, *, name_prefix: str = ""):
        self.w = jnp.ones(d_model)
        self.b = jnp.zeros(d_model)
        self.eps = eps
        self.hook_scale = HookPoint(name=f"{name_prefix}hook_scale")
        self.hook_normalized = HookPoint(name=f"{name_prefix}hook_normalized")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        # x: [seq_len, d_model]
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        scale = jnp.sqrt(var + self.eps)
        scale = self.hook_scale(scale, hook_state)
        normalized = (x - mean) / scale
        normalized = self.hook_normalized(normalized, hook_state)
        return normalized * self.w + self.b


class LayerNormPre(eqx.Module):
    """Layer Normalization without learned parameters (pre-norm, no affine).

    Just centers and normalizes. Used as a preprocessing step.
    """

    eps: float = eqx.field(static=True)
    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, d_model: int = 0, eps: float = 1e-5, *, name_prefix: str = ""):
        self.eps = eps
        self.hook_scale = HookPoint(name=f"{name_prefix}hook_scale")
        self.hook_normalized = HookPoint(name=f"{name_prefix}hook_normalized")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        scale = jnp.sqrt(var + self.eps)
        scale = self.hook_scale(scale, hook_state)
        normalized = (x - mean) / scale
        normalized = self.hook_normalized(normalized, hook_state)
        return normalized


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization with learned scale.

    Used by LLaMA/Mistral. No mean centering, no bias.
    """

    w: jnp.ndarray  # [d_model]
    eps: float = eqx.field(static=True)
    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, d_model: int, eps: float = 1e-5, *, name_prefix: str = ""):
        self.w = jnp.ones(d_model)
        self.eps = eps
        self.hook_scale = HookPoint(name=f"{name_prefix}hook_scale")
        self.hook_normalized = HookPoint(name=f"{name_prefix}hook_normalized")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        scale = self.hook_scale(rms, hook_state)
        normalized = x / scale
        normalized = self.hook_normalized(normalized, hook_state)
        return normalized * self.w


class RMSNormPre(eqx.Module):
    """RMS Normalization without learned parameters."""

    eps: float = eqx.field(static=True)
    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, d_model: int = 0, eps: float = 1e-5, *, name_prefix: str = ""):
        self.eps = eps
        self.hook_scale = HookPoint(name=f"{name_prefix}hook_scale")
        self.hook_normalized = HookPoint(name=f"{name_prefix}hook_normalized")

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        scale = self.hook_scale(rms, hook_state)
        normalized = x / scale
        normalized = self.hook_normalized(normalized, hook_state)
        return normalized


def get_layer_norm(
    normalization_type: str | None, d_model: int, eps: float = 1e-5, name_prefix: str = ""
) -> eqx.Module | None:
    """Factory function to create the appropriate normalization layer."""
    if normalization_type is None:
        return None
    norm_map = {
        "LN": LayerNorm,
        "LNPre": LayerNormPre,
        "RMS": RMSNorm,
        "RMSPre": RMSNormPre,
    }
    cls = norm_map.get(normalization_type)
    if cls is None:
        raise ValueError(f"Unknown normalization_type: {normalization_type}")
    return cls(d_model=d_model, eps=eps, name_prefix=name_prefix)
