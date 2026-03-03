"""Tests for HookState and HookPoint."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from irtk.hook_points import HookState, HookPoint, HookFn


class TestHookPoint:
    def test_identity_without_hook_state(self):
        """HookPoint is identity when hook_state is None."""
        hp = HookPoint(name="test")
        x = jnp.ones((5, 3))
        result = hp(x)
        assert jnp.array_equal(result, x)

    def test_identity_with_none_hook_state(self):
        hp = HookPoint(name="test")
        x = jnp.ones((5, 3))
        result = hp(x, hook_state=None)
        assert jnp.array_equal(result, x)

    def test_caching(self):
        """HookPoint caches activations when cache dict is provided."""
        hp = HookPoint(name="my_hook")
        cache = {}
        hs = HookState(cache=cache)
        x = jnp.array([1.0, 2.0, 3.0])
        result = hp(x, hs)
        assert jnp.array_equal(result, x)
        assert "my_hook" in cache
        assert jnp.array_equal(cache["my_hook"], x)

    def test_no_caching_when_cache_is_none(self):
        """No caching when cache is None (only hooks)."""
        hp = HookPoint(name="my_hook")
        hs = HookState(cache=None)
        x = jnp.array([1.0, 2.0, 3.0])
        result = hp(x, hs)
        assert jnp.array_equal(result, x)

    def test_hook_fn_modifies_activation(self):
        """Hook function can modify the activation."""
        hp = HookPoint(name="my_hook")

        def double_hook(x, name):
            return x * 2

        hs = HookState(hook_fns={"my_hook": double_hook})
        x = jnp.array([1.0, 2.0, 3.0])
        result = hp(x, hs)
        assert jnp.allclose(result, x * 2)

    def test_hook_fn_returning_none_keeps_original(self):
        """Hook returning None leaves activation unchanged."""
        hp = HookPoint(name="my_hook")
        called = []

        def logging_hook(x, name):
            called.append(name)
            return None

        hs = HookState(hook_fns={"my_hook": logging_hook})
        x = jnp.array([1.0, 2.0])
        result = hp(x, hs)
        assert jnp.array_equal(result, x)
        assert called == ["my_hook"]

    def test_hook_fn_not_called_for_wrong_name(self):
        """Hook function only called when name matches."""
        hp = HookPoint(name="my_hook")
        called = []

        def spy_hook(x, name):
            called.append(name)
            return None

        hs = HookState(hook_fns={"other_hook": spy_hook})
        x = jnp.array([1.0])
        hp(x, hs)
        assert called == []

    def test_cache_and_hook_together(self):
        """Caching happens before hook is applied; cache stores original value."""
        hp = HookPoint(name="my_hook")
        cache = {}

        def double_hook(x, name):
            return x * 2

        hs = HookState(hook_fns={"my_hook": double_hook}, cache=cache)
        x = jnp.array([1.0, 2.0])
        result = hp(x, hs)
        # Result is doubled
        assert jnp.allclose(result, x * 2)
        # Cache stores original (pre-hook) value
        assert jnp.array_equal(cache["my_hook"], x)

    def test_hook_point_is_eqx_module(self):
        """HookPoint is an Equinox module (works with pytree operations)."""
        hp = HookPoint(name="test")
        assert isinstance(hp, eqx.Module)
        # Should be flattenable
        leaves, treedef = jax.tree.flatten(hp)

    def test_jit_without_hook_state(self):
        """HookPoint works under JIT when hook_state is None."""
        hp = HookPoint(name="test")
        jitted = eqx.filter_jit(hp)
        x = jnp.ones((3, 4))
        result = jitted(x)
        assert jnp.array_equal(result, x)


class TestHookState:
    def test_default_construction(self):
        hs = HookState()
        assert hs.hook_fns == {}
        assert hs.cache is None

    def test_with_cache(self):
        cache = {}
        hs = HookState(cache=cache)
        assert hs.cache is cache

    def test_with_hook_fns(self):
        def my_fn(x, name):
            return x

        hs = HookState(hook_fns={"test": my_fn})
        assert "test" in hs.hook_fns
