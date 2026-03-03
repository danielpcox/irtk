"""Activation functions used in transformer models."""

import jax
import jax.numpy as jnp


def gelu_new(x: jnp.ndarray) -> jnp.ndarray:
    """GPT-2's approximate GELU (tanh approximation)."""
    return (
        0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))
    )


def solu(x: jnp.ndarray) -> jnp.ndarray:
    """SoLU activation: x * softmax(x) along last axis."""
    return x * jax.nn.softmax(x, axis=-1)


ACTIVATION_FN_DICT: dict = {
    "gelu": jax.nn.gelu,
    "gelu_new": gelu_new,
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "solu": solu,
}
