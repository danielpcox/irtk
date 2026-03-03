"""Residual stream attribution.

Trace how each component builds the final residual: per-position attribution,
directional decomposition, component overlap, and cumulative buildup tracking.
"""

import jax
import jax.numpy as jnp


def per_position_attribution(model, tokens, position=-1):
    """Attribute the residual stream at a position to each component.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with per-component attribution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1

    components = []

    # Embedding contribution
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    components.append({
        'name': 'embed',
        'layer': -1,
        'type': 'embed',
        'norm': float(jnp.linalg.norm(embed)),
        'vector': embed,
    })

    for l in range(n_layers):
        # Attention
        attn_out = cache[f'blocks.{l}.hook_attn_out'][pos]
        components.append({
            'name': f'L{l}_attn',
            'layer': l,
            'type': 'attn',
            'norm': float(jnp.linalg.norm(attn_out)),
            'vector': attn_out,
        })

        # MLP
        mlp_out = cache[f'blocks.{l}.hook_mlp_out'][pos]
        components.append({
            'name': f'L{l}_mlp',
            'layer': l,
            'type': 'mlp',
            'norm': float(jnp.linalg.norm(mlp_out)),
            'vector': mlp_out,
        })

    # Sort by norm
    components.sort(key=lambda c: -c['norm'])

    # Remove vectors from output (keep for internal use but don't return)
    result_components = [{k: v for k, v in c.items() if k != 'vector'} for c in components]

    total_norm = float(jnp.linalg.norm(cache[f'blocks.{n_layers-1}.hook_resid_post'][pos]))
    return {
        'position': pos,
        'per_component': result_components,
        'total_residual_norm': total_norm,
        'largest_contributor': result_components[0] if result_components else None,
    }


def directional_attribution(model, tokens, direction, position=-1):
    """Attribute a specific direction in the residual to each component.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        direction: direction vector [d_model]
        position: position to analyze

    Returns:
        dict with per-component directional attribution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    components = []

    # Embedding
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    proj = float(embed @ direction)
    components.append({
        'name': 'embed',
        'layer': -1,
        'projection': proj,
        'abs_projection': abs(proj),
    })

    for l in range(n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out'][pos]
        proj = float(attn_out @ direction)
        components.append({
            'name': f'L{l}_attn',
            'layer': l,
            'projection': proj,
            'abs_projection': abs(proj),
        })

        mlp_out = cache[f'blocks.{l}.hook_mlp_out'][pos]
        proj = float(mlp_out @ direction)
        components.append({
            'name': f'L{l}_mlp',
            'layer': l,
            'projection': proj,
            'abs_projection': abs(proj),
        })

    components.sort(key=lambda c: -c['abs_projection'])
    total_proj = float(cache[f'blocks.{n_layers-1}.hook_resid_post'][pos] @ direction)

    return {
        'position': pos,
        'per_component': components,
        'total_projection': total_proj,
        'sum_projections': sum(c['projection'] for c in components),
    }


def component_overlap_matrix(model, tokens, position=-1):
    """Compute pairwise cosine overlap between component contributions.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with overlap matrix.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1

    vectors = []
    names = []

    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    vectors.append(embed)
    names.append('embed')

    for l in range(n_layers):
        vectors.append(cache[f'blocks.{l}.hook_attn_out'][pos])
        names.append(f'L{l}_attn')
        vectors.append(cache[f'blocks.{l}.hook_mlp_out'][pos])
        names.append(f'L{l}_mlp')

    n = len(vectors)
    overlap = jnp.zeros((n, n))
    norms = [jnp.linalg.norm(v) + 1e-10 for v in vectors]

    for i in range(n):
        for j in range(n):
            cos = float(jnp.sum(vectors[i] * vectors[j]) / (norms[i] * norms[j]))
            overlap = overlap.at[i, j].set(cos)

    # Mean off-diagonal overlap
    mask = 1.0 - jnp.eye(n)
    mean_overlap = float(jnp.mean(jnp.abs(overlap) * mask))

    return {
        'names': names,
        'overlap_matrix': overlap,
        'mean_overlap': mean_overlap,
        'n_components': n,
    }


def cumulative_buildup(model, tokens, position=-1):
    """Track how the residual builds up layer by layer.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with cumulative buildup per layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    stages = []
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post'][pos]
        norm = float(jnp.linalg.norm(resid))
        logits = resid @ W_U + b_U
        top_token = int(jnp.argmax(logits))
        confidence = float(jax.nn.softmax(logits)[top_token])

        stages.append({
            'layer': l,
            'residual_norm': norm,
            'top_prediction': top_token,
            'confidence': confidence,
        })

    # Norm growth rate
    norms = [s['residual_norm'] for s in stages]
    if len(norms) >= 2:
        growth_rate = (norms[-1] - norms[0]) / max(len(norms) - 1, 1)
    else:
        growth_rate = 0.0

    return {
        'position': pos,
        'stages': stages,
        'norm_growth_rate': growth_rate,
        'final_prediction': stages[-1]['top_prediction'] if stages else 0,
    }


def logit_attribution_by_component(model, tokens, position=-1):
    """Attribute final logits to each component's contribution.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with per-component logit attribution for the top predicted token.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    pos = position if position >= 0 else len(tokens) - 1
    W_U = model.unembed.W_U

    # Get target token
    logits = model(tokens)
    target = int(jnp.argmax(logits[pos]))
    target_direction = W_U[:, target]

    components = []

    # Embedding
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    logit = float(embed @ target_direction)
    components.append({
        'name': 'embed',
        'logit_contribution': logit,
        'abs_contribution': abs(logit),
    })

    for l in range(n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out'][pos]
        logit = float(attn_out @ target_direction)
        components.append({
            'name': f'L{l}_attn',
            'logit_contribution': logit,
            'abs_contribution': abs(logit),
        })

        mlp_out = cache[f'blocks.{l}.hook_mlp_out'][pos]
        logit = float(mlp_out @ target_direction)
        components.append({
            'name': f'L{l}_mlp',
            'logit_contribution': logit,
            'abs_contribution': abs(logit),
        })

    components.sort(key=lambda c: -c['abs_contribution'])
    return {
        'position': pos,
        'target_token': target,
        'per_component': components,
        'total_logit': sum(c['logit_contribution'] for c in components),
    }
