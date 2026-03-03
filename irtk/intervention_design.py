"""Intervention design: targeted activation interventions and their effects."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def scale_component_effect(model: HookedTransformer, tokens: jnp.ndarray, component: str, scale: float) -> dict:
    """Scale a specific component's output and measure the effect.

    component: 'L{layer}_attn' or 'L{layer}_mlp'
    """
    logits_clean = model(tokens)
    clean_probs = jax.nn.softmax(logits_clean[-1])
    clean_top = int(jnp.argmax(clean_probs))
    clean_conf = float(clean_probs[clean_top])

    # Parse component
    parts = component.split('_')
    layer = int(parts[0][1:])
    comp_type = parts[1]

    if comp_type == 'attn':
        hook_name = f'blocks.{layer}.hook_attn_out'
    else:
        hook_name = f'blocks.{layer}.hook_mlp_out'

    def scale_hook(x, name):
        return x * scale

    hooks = [(hook_name, scale_hook)]
    logits_scaled = model.run_with_hooks(tokens, fwd_hooks=hooks)
    scaled_probs = jax.nn.softmax(logits_scaled[-1])
    scaled_top = int(jnp.argmax(scaled_probs))
    scaled_conf = float(scaled_probs[scaled_top])

    # KL divergence
    kl = float(jnp.sum(clean_probs * (jnp.log(clean_probs + 1e-10) - jnp.log(scaled_probs + 1e-10))))

    return {
        'component': component,
        'scale': scale,
        'clean_prediction': clean_top,
        'clean_confidence': clean_conf,
        'scaled_prediction': scaled_top,
        'scaled_confidence': scaled_conf,
        'prediction_changed': clean_top != scaled_top,
        'kl_divergence': kl,
    }


def add_direction_effect(model: HookedTransformer, tokens: jnp.ndarray, layer: int, direction: jnp.ndarray, magnitude: float = 1.0, position: int = -1) -> dict:
    """Add a direction to the residual stream and measure the effect.

    Useful for steering experiments.
    """
    logits_clean = model(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    clean_probs = jax.nn.softmax(logits_clean[pos])
    clean_top = int(jnp.argmax(clean_probs))
    clean_conf = float(clean_probs[clean_top])

    direction_normed = direction / (jnp.linalg.norm(direction) + 1e-10)
    steering_vec = direction_normed * magnitude

    def add_hook(x, name):
        modified = x.at[pos].add(steering_vec)
        return modified

    hook_name = f'blocks.{layer}.hook_resid_post'
    hooks = [(hook_name, add_hook)]
    logits_steered = model.run_with_hooks(tokens, fwd_hooks=hooks)

    steered_probs = jax.nn.softmax(logits_steered[pos])
    steered_top = int(jnp.argmax(steered_probs))
    steered_conf = float(steered_probs[steered_top])

    return {
        'layer': layer,
        'position': pos,
        'magnitude': magnitude,
        'clean_prediction': clean_top,
        'clean_confidence': clean_conf,
        'steered_prediction': steered_top,
        'steered_confidence': steered_conf,
        'prediction_changed': clean_top != steered_top,
    }


def zero_ablation_sweep(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Zero-ablate each component individually and rank by impact.

    Returns sorted list of components by their ablation effect.
    """
    logits_clean = model(tokens)
    clean_probs = jax.nn.softmax(logits_clean[-1])
    n_layers = model.cfg.n_layers

    results = []
    for layer in range(n_layers):
        for comp_type, hook_name in [
            ('attn', f'blocks.{layer}.hook_attn_out'),
            ('mlp', f'blocks.{layer}.hook_mlp_out'),
        ]:
            def zero_hook(x, name):
                return jnp.zeros_like(x)

            hooks = [(hook_name, zero_hook)]
            logits_ablated = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_probs = jax.nn.softmax(logits_ablated[-1])

            kl = float(jnp.sum(clean_probs * (jnp.log(clean_probs + 1e-10) - jnp.log(ablated_probs + 1e-10))))

            results.append({
                'component': f'L{layer}_{comp_type}',
                'layer': layer,
                'type': comp_type,
                'kl_divergence': kl,
                'prediction_changed': int(jnp.argmax(clean_probs)) != int(jnp.argmax(ablated_probs)),
            })

    results.sort(key=lambda x: x['kl_divergence'], reverse=True)

    return {
        'per_component': results,
        'most_important': results[0]['component'] if results else '',
        'n_prediction_changers': sum(1 for r in results if r['prediction_changed']),
    }


def mean_ablation_effect(model: HookedTransformer, tokens: jnp.ndarray, layer: int, component: str = 'attn') -> dict:
    """Replace a component's output with its mean and measure the effect.

    Less destructive than zero ablation — preserves the average contribution.
    """
    _, cache = model.run_with_cache(tokens)

    if component == 'attn':
        hook_name = f'blocks.{layer}.hook_attn_out'
    else:
        hook_name = f'blocks.{layer}.hook_mlp_out'

    comp_output = cache[hook_name]  # [seq, d_model]
    mean_output = jnp.mean(comp_output, axis=0, keepdims=True)  # [1, d_model]

    logits_clean = model(tokens)
    clean_probs = jax.nn.softmax(logits_clean[-1])

    def mean_hook(x, name):
        return jnp.broadcast_to(mean_output, x.shape)

    hooks = [(hook_name, mean_hook)]
    logits_mean = model.run_with_hooks(tokens, fwd_hooks=hooks)
    mean_probs = jax.nn.softmax(logits_mean[-1])

    kl = float(jnp.sum(clean_probs * (jnp.log(clean_probs + 1e-10) - jnp.log(mean_probs + 1e-10))))

    return {
        'layer': layer,
        'component': component,
        'kl_divergence': kl,
        'clean_prediction': int(jnp.argmax(clean_probs)),
        'mean_prediction': int(jnp.argmax(mean_probs)),
        'prediction_changed': int(jnp.argmax(clean_probs)) != int(jnp.argmax(mean_probs)),
    }


def progressive_ablation(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Progressively ablate layers from last to first, measuring cumulative damage.

    Shows how much context the model needs for its prediction.
    """
    logits_clean = model(tokens)
    clean_probs = jax.nn.softmax(logits_clean[-1])
    clean_top = int(jnp.argmax(clean_probs))
    n_layers = model.cfg.n_layers

    per_step = []
    ablated_layers = set()

    for layer in range(n_layers - 1, -1, -1):
        ablated_layers.add(layer)

        hooks = []
        for l in ablated_layers:
            def make_zero(x, name):
                return jnp.zeros_like(x)
            hooks.append((f'blocks.{l}.hook_attn_out', make_zero))
            hooks.append((f'blocks.{l}.hook_mlp_out', make_zero))

        logits_ablated = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_probs = jax.nn.softmax(logits_ablated[-1])

        kl = float(jnp.sum(clean_probs * (jnp.log(clean_probs + 1e-10) - jnp.log(ablated_probs + 1e-10))))
        top = int(jnp.argmax(ablated_probs))

        per_step.append({
            'n_ablated': len(ablated_layers),
            'last_ablated': layer,
            'kl_divergence': kl,
            'prediction': top,
            'prediction_intact': top == clean_top,
        })

    n_needed = n_layers
    for step in per_step:
        if not step['prediction_intact']:
            n_needed = n_layers - step['n_ablated'] + 1
            break

    return {
        'per_step': per_step,
        'clean_prediction': clean_top,
        'min_layers_needed': n_needed,
    }
