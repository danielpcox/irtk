"""Residual stream surgery.

Surgical modifications to the residual stream: project out directions,
clamp dimensions, add/remove component contributions, and measure effects.
"""

import jax
import jax.numpy as jnp


def project_out_direction(model, tokens, direction, layer):
    """Remove a direction from the residual stream at a given layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        direction: [d_model] direction to project out
        layer: layer at which to intervene

    Returns:
        dict with logit changes and projection magnitude.
    """
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    # Clean run
    clean_logits = model(tokens)

    # Intervention: project out direction from residual at hook_resid_pre
    def hook_fn(x, name):
        # x: [seq, d_model]
        proj = jnp.sum(x * direction, axis=-1, keepdims=True)  # [seq, 1]
        return x - proj * direction

    hook_name = f'blocks.{layer}.hook_resid_pre'
    modified_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    # Measure effect
    logit_diff = modified_logits - clean_logits
    max_change = float(jnp.max(jnp.abs(logit_diff)))

    # How much was projected out?
    _, cache = model.run_with_cache(tokens)
    resid = cache[hook_name]  # [seq, d_model]
    projections = jnp.sum(resid * direction, axis=-1)  # [seq]

    return {
        'layer': layer,
        'max_logit_change': max_change,
        'mean_logit_change': float(jnp.mean(jnp.abs(logit_diff))),
        'projection_magnitudes': [float(p) for p in projections],
        'mean_projection': float(jnp.mean(jnp.abs(projections))),
    }


def clamp_residual_norm(model, tokens, layer, max_norm):
    """Clamp the residual stream norm at a given layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer at which to intervene
        max_norm: maximum allowed norm

    Returns:
        dict with effect of clamping.
    """
    clean_logits = model(tokens)

    def hook_fn(x, name):
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-10))
        return x * scale

    hook_name = f'blocks.{layer}.hook_resid_pre'
    modified_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    # Measure original norms
    _, cache = model.run_with_cache(tokens)
    resid = cache[hook_name]
    norms = jnp.linalg.norm(resid, axis=-1)  # [seq]
    n_clamped = int(jnp.sum(norms > max_norm))

    logit_diff = modified_logits - clean_logits

    return {
        'layer': layer,
        'max_norm': max_norm,
        'n_clamped': n_clamped,
        'n_total': int(norms.shape[0]),
        'original_mean_norm': float(jnp.mean(norms)),
        'original_max_norm': float(jnp.max(norms)),
        'max_logit_change': float(jnp.max(jnp.abs(logit_diff))),
        'mean_logit_change': float(jnp.mean(jnp.abs(logit_diff))),
    }


def remove_component_contribution(model, tokens, layer, component='attn'):
    """Remove a component's contribution from a specific layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        component: 'attn' or 'mlp'

    Returns:
        dict with effect of removing the component.
    """
    clean_logits = model(tokens)

    if component == 'attn':
        hook_name = f'blocks.{layer}.hook_attn_out'
    else:
        hook_name = f'blocks.{layer}.hook_mlp_out'

    def hook_fn(x, name):
        return jnp.zeros_like(x)

    modified_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    logit_diff = modified_logits - clean_logits

    # Get original component output norm
    _, cache = model.run_with_cache(tokens)
    comp_output = cache[hook_name]
    comp_norm = float(jnp.mean(jnp.linalg.norm(comp_output, axis=-1)))

    # KL divergence
    clean_probs = jax.nn.softmax(clean_logits, axis=-1)
    mod_probs = jax.nn.softmax(modified_logits, axis=-1)
    kl = float(jnp.mean(jnp.sum(clean_probs * jnp.log(jnp.maximum(clean_probs, 1e-10) /
                                                         jnp.maximum(mod_probs, 1e-10)), axis=-1)))

    return {
        'layer': layer,
        'component': component,
        'component_norm': comp_norm,
        'kl_divergence': kl,
        'max_logit_change': float(jnp.max(jnp.abs(logit_diff))),
        'mean_logit_change': float(jnp.mean(jnp.abs(logit_diff))),
    }


def add_steering_at_layer(model, tokens, direction, layer, alpha=1.0):
    """Add a steering vector to the residual stream at a given layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        direction: [d_model] direction to add
        layer: layer at which to intervene
        alpha: scaling factor

    Returns:
        dict with effect of steering.
    """
    clean_logits = model(tokens)

    def hook_fn(x, name):
        return x + alpha * direction

    hook_name = f'blocks.{layer}.hook_resid_pre'
    modified_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    logit_diff = modified_logits - clean_logits

    # Top promoted/demoted tokens at last position
    diff_last = logit_diff[-1]  # [d_vocab]
    top_promoted = jnp.argsort(-diff_last)[:5]
    top_demoted = jnp.argsort(diff_last)[:5]

    return {
        'layer': layer,
        'alpha': alpha,
        'max_logit_change': float(jnp.max(jnp.abs(logit_diff))),
        'mean_logit_change': float(jnp.mean(jnp.abs(logit_diff))),
        'top_promoted': [{'token': int(t), 'logit_change': float(diff_last[t])} for t in top_promoted],
        'top_demoted': [{'token': int(t), 'logit_change': float(diff_last[t])} for t in top_demoted],
    }


def dimension_clamping(model, tokens, layer, dimensions, value=0.0):
    """Clamp specific dimensions of the residual stream to a fixed value.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer at which to intervene
        dimensions: list of dimension indices to clamp
        value: value to clamp to

    Returns:
        dict with effect of dimension clamping.
    """
    clean_logits = model(tokens)

    dims = jnp.array(dimensions)

    def hook_fn(x, name):
        for d in dimensions:
            x = x.at[:, d].set(value)
        return x

    hook_name = f'blocks.{layer}.hook_resid_pre'
    modified_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    logit_diff = modified_logits - clean_logits

    # Get original values of clamped dimensions
    _, cache = model.run_with_cache(tokens)
    resid = cache[hook_name]
    original_values = [float(jnp.mean(jnp.abs(resid[:, d]))) for d in dimensions]

    return {
        'layer': layer,
        'dimensions': dimensions,
        'clamp_value': value,
        'original_mean_abs_values': original_values,
        'max_logit_change': float(jnp.max(jnp.abs(logit_diff))),
        'mean_logit_change': float(jnp.mean(jnp.abs(logit_diff))),
        'n_dims_clamped': len(dimensions),
        'fraction_clamped': len(dimensions) / model.cfg.d_model,
    }
