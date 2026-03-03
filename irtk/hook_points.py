"""Hook system for activation inspection and intervention.

HookState is a plain Python object (NOT a JAX PyTree) threaded through every
component's __call__. When None, hooks are pure identity with zero overhead.
"""

from typing import Callable, Optional, Union

import jax.numpy as jnp
import equinox as eqx


# Hook function signature: (activation, hook_name) -> Optional[activation]
# Return None to leave activation unchanged, or return modified activation.
HookFn = Callable[[jnp.ndarray, str], Optional[jnp.ndarray]]


class HookState:
    """Mutable state threaded through model forward pass for activation caching/intervention.

    This is a plain Python object, NOT a JAX PyTree. It enables mutation
    (caching activations) which is incompatible with JIT. For interpretability
    use outside JIT.
    """

    def __init__(
        self,
        hook_fns: Optional[dict[str, HookFn]] = None,
        cache: Optional[dict[str, jnp.ndarray]] = None,
    ):
        self.hook_fns = hook_fns if hook_fns is not None else {}
        self.cache = cache  # None means no caching; {} means cache everything


class HookPoint(eqx.Module):
    """A named point in the model where activations can be inspected or modified.

    When hook_state is None, this is a pure identity function with zero overhead.
    When hook_state is provided, it caches the activation and/or applies a hook function.
    """

    name: str = eqx.field(static=True)

    def __call__(
        self, x: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        if hook_state is None:
            return x
        if hook_state.cache is not None:
            hook_state.cache[self.name] = x
        fn = hook_state.hook_fns.get(self.name)
        if fn is not None:
            result = fn(x, self.name)
            if result is not None:
                x = result
        return x
