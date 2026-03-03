"""Activation surgery.

Surgical modifications to activations: clamping, scaling, rotation,
projection onto/off subspaces, and targeted replacement.

References:
    Turner et al. (2023) "Activation Addition: Steering Language Models Without Optimization"
    Li et al. (2024) "Inference-Time Intervention"
"""

import jax
import jax.numpy as jnp
import numpy as np


def clamp_activation(model, tokens, hook_name, pos, min_val=None, max_val=None):
    """Clamp activations at a hook point within bounds.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        hook_name: Hook point to clamp.
        pos: Position(s) to clamp (int or list).
        min_val: Minimum value (None = no lower bound).
        max_val: Maximum value (None = no upper bound).

    Returns:
        dict with:
            original_logits: logits without clamping
            clamped_logits: logits with clamping
            logit_diff: max absolute logit change
            n_clamped_values: number of values affected
    """
    from irtk.hook_points import HookState

    original_logits = model(tokens)

    # Run with cache to get original activation
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    orig_act = cache_state.cache.get(hook_name)

    n_clamped = 0
    if orig_act is not None:
        positions = [pos] if isinstance(pos, int) else pos
        for p in positions:
            vals = np.array(orig_act[p])
            if min_val is not None:
                n_clamped += int(np.sum(vals < min_val))
            if max_val is not None:
                n_clamped += int(np.sum(vals > max_val))

    def clamp_fn(x, name):
        result = x
        positions = [pos] if isinstance(pos, int) else pos
        for p in positions:
            val = result[p]
            if min_val is not None:
                val = jnp.maximum(val, min_val)
            if max_val is not None:
                val = jnp.minimum(val, max_val)
            result = result.at[p].set(val)
        return result

    state = HookState(hook_fns={hook_name: clamp_fn}, cache={})
    clamped_logits = model(tokens, hook_state=state)

    diff = float(jnp.max(jnp.abs(original_logits - clamped_logits)))

    return {
        "original_logits": np.array(original_logits),
        "clamped_logits": np.array(clamped_logits),
        "logit_diff": diff,
        "n_clamped_values": n_clamped,
    }


def scale_activation(model, tokens, hook_name, pos, scale_factor=2.0):
    """Scale activations at a hook point by a factor.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        hook_name: Hook point to scale.
        pos: Position to scale.
        scale_factor: Multiplicative factor.

    Returns:
        dict with:
            original_logits: logits without scaling
            scaled_logits: logits with scaling
            logit_diff: max absolute logit change
            original_norm: norm of original activation at pos
            scaled_norm: norm of scaled activation
    """
    from irtk.hook_points import HookState

    # Get original activation
    cache_state = HookState(hook_fns={}, cache={})
    original_logits = model(tokens, hook_state=cache_state)
    orig_act = cache_state.cache.get(hook_name)
    orig_norm = float(jnp.linalg.norm(orig_act[pos])) if orig_act is not None else 0.0

    def scale_fn(x, name):
        return x.at[pos].set(x[pos] * scale_factor)

    state = HookState(hook_fns={hook_name: scale_fn}, cache={})
    scaled_logits = model(tokens, hook_state=state)

    diff = float(jnp.max(jnp.abs(original_logits - scaled_logits)))

    return {
        "original_logits": np.array(original_logits),
        "scaled_logits": np.array(scaled_logits),
        "logit_diff": diff,
        "original_norm": orig_norm,
        "scaled_norm": orig_norm * abs(scale_factor),
    }


def project_activation(model, tokens, hook_name, pos, direction, mode="onto"):
    """Project activation onto or off a direction.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        hook_name: Hook point to modify.
        pos: Position to project.
        direction: Direction vector [d_model].
        mode: "onto" to keep only the component along direction,
              "off" to remove the component along direction.

    Returns:
        dict with:
            original_logits: logits without projection
            projected_logits: logits with projection
            logit_diff: max absolute logit change
            component_magnitude: magnitude of activation along direction
            fraction_removed: fraction of activation norm removed
    """
    from irtk.hook_points import HookState

    direction = jnp.array(direction)
    dir_norm = jnp.linalg.norm(direction) + 1e-10
    dir_unit = direction / dir_norm

    # Get original activation
    cache_state = HookState(hook_fns={}, cache={})
    original_logits = model(tokens, hook_state=cache_state)
    orig_act = cache_state.cache.get(hook_name)

    component_mag = 0.0
    frac_removed = 0.0
    if orig_act is not None:
        act_vec = orig_act[pos]
        comp = float(jnp.dot(act_vec, dir_unit))
        component_mag = abs(comp)
        act_norm = float(jnp.linalg.norm(act_vec)) + 1e-10
        if mode == "off":
            frac_removed = component_mag / act_norm
        else:
            frac_removed = 1.0 - component_mag / act_norm

    def project_fn(x, name):
        act = x[pos]
        comp = jnp.dot(act, dir_unit)
        if mode == "onto":
            new_act = comp * dir_unit
        else:
            new_act = act - comp * dir_unit
        return x.at[pos].set(new_act)

    state = HookState(hook_fns={hook_name: project_fn}, cache={})
    projected_logits = model(tokens, hook_state=state)

    diff = float(jnp.max(jnp.abs(original_logits - projected_logits)))

    return {
        "original_logits": np.array(original_logits),
        "projected_logits": np.array(projected_logits),
        "logit_diff": diff,
        "component_magnitude": component_mag,
        "fraction_removed": float(frac_removed),
    }


def rotate_activation(model, tokens, hook_name, pos, from_dir, to_dir, strength=1.0):
    """Rotate the component of activation from one direction to another.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        hook_name: Hook point to modify.
        pos: Position to rotate.
        from_dir: Source direction [d_model].
        to_dir: Target direction [d_model].
        strength: Rotation strength (0=none, 1=full rotation).

    Returns:
        dict with:
            original_logits: logits without rotation
            rotated_logits: logits with rotation
            logit_diff: max absolute logit change
            rotation_angle: angle between from_dir and to_dir
            component_in_from: magnitude along from_dir before
    """
    from irtk.hook_points import HookState

    from_dir = jnp.array(from_dir)
    to_dir = jnp.array(to_dir)
    from_unit = from_dir / (jnp.linalg.norm(from_dir) + 1e-10)
    to_unit = to_dir / (jnp.linalg.norm(to_dir) + 1e-10)

    angle = float(jnp.arccos(jnp.clip(jnp.dot(from_unit, to_unit), -1, 1)))

    # Get original
    cache_state = HookState(hook_fns={}, cache={})
    original_logits = model(tokens, hook_state=cache_state)
    orig_act = cache_state.cache.get(hook_name)
    comp_from = 0.0
    if orig_act is not None:
        comp_from = float(jnp.abs(jnp.dot(orig_act[pos], from_unit)))

    def rotate_fn(x, name):
        act = x[pos]
        comp = jnp.dot(act, from_unit)
        # Remove from_dir component, add to_dir component
        new_act = act - strength * comp * from_unit + strength * comp * to_unit
        return x.at[pos].set(new_act)

    state = HookState(hook_fns={hook_name: rotate_fn}, cache={})
    rotated_logits = model(tokens, hook_state=state)

    diff = float(jnp.max(jnp.abs(original_logits - rotated_logits)))

    return {
        "original_logits": np.array(original_logits),
        "rotated_logits": np.array(rotated_logits),
        "logit_diff": diff,
        "rotation_angle": float(angle),
        "component_in_from": comp_from,
    }


def targeted_replacement(model, tokens, hook_name, pos, replacement_value):
    """Replace activation at a specific position with a given value.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        hook_name: Hook point to modify.
        pos: Position to replace.
        replacement_value: Value to insert [d_model] or scalar.

    Returns:
        dict with:
            original_logits: logits without replacement
            replaced_logits: logits with replacement
            logit_diff: max absolute logit change
            displacement: L2 distance between original and replacement
    """
    from irtk.hook_points import HookState

    replacement = jnp.array(replacement_value)

    # Get original
    cache_state = HookState(hook_fns={}, cache={})
    original_logits = model(tokens, hook_state=cache_state)
    orig_act = cache_state.cache.get(hook_name)

    displacement = 0.0
    if orig_act is not None:
        displacement = float(jnp.linalg.norm(orig_act[pos] - replacement))

    def replace_fn(x, name):
        return x.at[pos].set(replacement)

    state = HookState(hook_fns={hook_name: replace_fn}, cache={})
    replaced_logits = model(tokens, hook_state=state)

    diff = float(jnp.max(jnp.abs(original_logits - replaced_logits)))

    return {
        "original_logits": np.array(original_logits),
        "replaced_logits": np.array(replaced_logits),
        "logit_diff": diff,
        "displacement": displacement,
    }
