"""Prediction uncertainty analysis.

Analyze model uncertainty at different levels: per-layer uncertainty
evolution, component contribution to uncertainty, uncertainty
decomposition, and calibration analysis.
"""

import jax
import jax.numpy as jnp


def layer_uncertainty_evolution(model, tokens):
    """Track uncertainty (entropy) at each layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with per-layer uncertainty.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']
        logits = resid @ W_U + b_U  # [seq, d_vocab]
        probs = jax.nn.softmax(logits, axis=-1)

        entropies = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)  # [seq]
        confidences = jnp.max(probs, axis=-1)

        per_layer.append({
            'layer': l,
            'mean_entropy': float(jnp.mean(entropies)),
            'mean_confidence': float(jnp.mean(confidences)),
            'min_confidence': float(jnp.min(confidences)),
            'max_confidence': float(jnp.max(confidences)),
        })

    # Uncertainty trend
    entropies = [p['mean_entropy'] for p in per_layer]
    if len(entropies) >= 2:
        trend = (entropies[-1] - entropies[0]) / max(len(entropies) - 1, 1)
    else:
        trend = 0.0

    return {
        'per_layer': per_layer,
        'entropy_trend': trend,
        'resolves_uncertainty': trend < 0,
    }


def component_uncertainty_contribution(model, tokens, position=-1):
    """Measure how each component affects prediction uncertainty.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with per-component uncertainty effects.
    """
    clean_logits = model(tokens)
    pos = position if position >= 0 else len(tokens) - 1
    n_layers = model.cfg.n_layers

    clean_probs = jax.nn.softmax(clean_logits[pos])
    clean_entropy = -float(jnp.sum(clean_probs * jnp.log(clean_probs + 1e-10)))

    components = []
    for l in range(n_layers):
        # Ablate attention
        def make_zero_hook():
            def hook_fn(x, name):
                return jnp.zeros_like(x)
            return hook_fn

        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(f'blocks.{l}.hook_attn_out', make_zero_hook())])
        mod_probs = jax.nn.softmax(mod_logits[pos])
        mod_entropy = -float(jnp.sum(mod_probs * jnp.log(mod_probs + 1e-10)))
        components.append({
            'name': f'L{l}_attn',
            'entropy_change': mod_entropy - clean_entropy,
            'reduces_uncertainty': mod_entropy > clean_entropy,
        })

        # Ablate MLP
        mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(f'blocks.{l}.hook_mlp_out', make_zero_hook())])
        mod_probs = jax.nn.softmax(mod_logits[pos])
        mod_entropy = -float(jnp.sum(mod_probs * jnp.log(mod_probs + 1e-10)))
        components.append({
            'name': f'L{l}_mlp',
            'entropy_change': mod_entropy - clean_entropy,
            'reduces_uncertainty': mod_entropy > clean_entropy,
        })

    components.sort(key=lambda c: -abs(c['entropy_change']))
    return {
        'position': pos,
        'clean_entropy': clean_entropy,
        'per_component': components,
        'most_uncertainty_reducing': next((c for c in components if c['reduces_uncertainty']), None),
    }


def uncertainty_decomposition(model, tokens):
    """Decompose uncertainty into aleatoric (data) vs epistemic (model) components.

    Uses the gap between top-1 and top-2 predictions as a proxy for model
    confidence, vs overall entropy for total uncertainty.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with uncertainty decomposition.
    """
    logits = model(tokens)
    probs = jax.nn.softmax(logits, axis=-1)

    per_position = []
    for pos in range(len(tokens)):
        p = probs[pos]
        entropy = -float(jnp.sum(p * jnp.log(p + 1e-10)))
        sorted_p = jnp.sort(p)[::-1]
        top1 = float(sorted_p[0])
        top2 = float(sorted_p[1])

        # Margin between top-1 and top-2 as confidence proxy
        margin = top1 - top2

        per_position.append({
            'position': pos,
            'total_entropy': entropy,
            'top1_probability': top1,
            'top2_probability': top2,
            'confidence_margin': margin,
            'is_uncertain': margin < 0.1,
        })

    return {
        'per_position': per_position,
        'mean_entropy': float(jnp.mean(jnp.array([p['total_entropy'] for p in per_position]))),
        'uncertain_fraction': sum(1 for p in per_position if p['is_uncertain']) / max(len(per_position), 1),
    }


def position_uncertainty_ranking(model, tokens):
    """Rank positions by prediction uncertainty.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with positions ranked by uncertainty.
    """
    logits = model(tokens)
    probs = jax.nn.softmax(logits, axis=-1)

    per_position = []
    for pos in range(len(tokens)):
        p = probs[pos]
        entropy = -float(jnp.sum(p * jnp.log(p + 1e-10)))
        confidence = float(jnp.max(p))
        n_plausible = int(jnp.sum(p > 0.05))

        per_position.append({
            'position': pos,
            'entropy': entropy,
            'confidence': confidence,
            'n_plausible_tokens': n_plausible,
        })

    per_position.sort(key=lambda p: -p['entropy'])
    return {
        'per_position': per_position,
        'most_uncertain': per_position[0]['position'] if per_position else 0,
        'most_certain': per_position[-1]['position'] if per_position else 0,
    }


def uncertainty_source_localization(model, tokens, position=-1):
    """Identify which layers contribute most to final uncertainty.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with per-layer uncertainty contribution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_entropy = None
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']
        logits = resid[pos] @ W_U + b_U
        probs = jax.nn.softmax(logits)
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))

        delta = entropy - prev_entropy if prev_entropy is not None else 0.0
        per_layer.append({
            'layer': l,
            'entropy': entropy,
            'entropy_delta': delta,
            'increases_uncertainty': delta > 0,
        })
        prev_entropy = entropy

    return {
        'position': pos,
        'per_layer': per_layer,
        'final_entropy': per_layer[-1]['entropy'] if per_layer else 0.0,
        'biggest_entropy_increase': max(per_layer, key=lambda p: p['entropy_delta'])['layer'] if per_layer else 0,
    }
