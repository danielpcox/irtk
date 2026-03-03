"""Residual norm decomposition: trace how each component contributes to the residual norm."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def norm_contribution_by_component(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How much does each component contribute to the residual stream norm?

    Decomposes ||residual||^2 into cross-term contributions.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    # Collect components
    components = []
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    components.append(('embed', embed))

    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]
        components.append((f'attn_{layer}', attn_out))
        components.append((f'mlp_{layer}', mlp_out))

    residual = sum(c[1] for c in components)
    total_norm_sq = float(jnp.sum(residual ** 2))

    per_component = []
    for name, vec in components:
        self_norm_sq = float(jnp.sum(vec ** 2))
        # Projection onto residual direction
        residual_dir = residual / (jnp.linalg.norm(residual) + 1e-10)
        projection = float(jnp.dot(vec, residual_dir))
        fraction = self_norm_sq / (total_norm_sq + 1e-10)

        per_component.append({
            'component': name,
            'self_norm': float(jnp.linalg.norm(vec)),
            'self_norm_sq': self_norm_sq,
            'fraction_of_total': fraction,
            'projection_onto_residual': projection,
        })

    return {
        'position': pos,
        'total_norm': float(jnp.linalg.norm(residual)),
        'per_component': per_component,
    }


def layerwise_norm_buildup(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How does the residual norm build up layer by layer?

    Tracks norm after each component addition.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    running = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    steps = [{'step': 'embed', 'norm': float(jnp.linalg.norm(running))}]

    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        running = running + attn_out
        steps.append({
            'step': f'attn_{layer}',
            'norm': float(jnp.linalg.norm(running)),
            'delta_norm': float(jnp.linalg.norm(attn_out)),
        })

        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]
        running = running + mlp_out
        steps.append({
            'step': f'mlp_{layer}',
            'norm': float(jnp.linalg.norm(running)),
            'delta_norm': float(jnp.linalg.norm(mlp_out)),
        })

    return {
        'position': pos,
        'steps': steps,
        'final_norm': float(jnp.linalg.norm(running)),
    }


def norm_direction_decomposition(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Decompose the residual direction into component contributions.

    Shows which components push the residual in its final direction.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    components = []
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    components.append(('embed', embed))
    for layer in range(n_layers):
        components.append((f'attn_{layer}', cache[f'blocks.{layer}.hook_attn_out'][pos]))
        components.append((f'mlp_{layer}', cache[f'blocks.{layer}.hook_mlp_out'][pos]))

    residual = sum(c[1] for c in components)
    residual_dir = residual / (jnp.linalg.norm(residual) + 1e-10)

    per_component = []
    for name, vec in components:
        proj = float(jnp.dot(vec, residual_dir))
        cos = float(jnp.dot(vec, residual_dir) / (jnp.linalg.norm(vec) + 1e-10))
        per_component.append({
            'component': name,
            'projection': proj,
            'cosine_with_residual': cos,
            'is_constructive': bool(proj > 0),
        })

    total_proj = sum(p['projection'] for p in per_component)
    return {
        'position': pos,
        'per_component': per_component,
        'total_projection': total_proj,
        'n_constructive': sum(1 for p in per_component if p['is_constructive']),
    }


def cross_position_norm_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How does the final residual norm vary across positions?

    Identifies positions with unusually large or small norms.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    n_layers = model.cfg.n_layers

    per_position = []
    norms = []
    for pos in range(seq_len):
        running = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
        for layer in range(n_layers):
            running = running + cache[f'blocks.{layer}.hook_attn_out'][pos]
            running = running + cache[f'blocks.{layer}.hook_mlp_out'][pos]
        norm = float(jnp.linalg.norm(running))
        norms.append(norm)

    mean_norm = sum(norms) / len(norms)
    std_norm = (sum((n - mean_norm) ** 2 for n in norms) / len(norms)) ** 0.5

    for pos, norm in enumerate(norms):
        per_position.append({
            'position': pos,
            'token': int(tokens[pos]),
            'norm': norm,
            'z_score': (norm - mean_norm) / (std_norm + 1e-10),
            'is_outlier': bool(abs(norm - mean_norm) > 2 * std_norm),
        })

    return {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'per_position': per_position,
        'n_outliers': sum(1 for p in per_position if p['is_outlier']),
    }


def component_interference_matrix(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Pairwise dot products between component outputs.

    Shows which components interfere constructively or destructively.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    names = ['embed']
    vecs = [cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]]
    for layer in range(n_layers):
        names.append(f'attn_{layer}')
        vecs.append(cache[f'blocks.{layer}.hook_attn_out'][pos])
        names.append(f'mlp_{layer}')
        vecs.append(cache[f'blocks.{layer}.hook_mlp_out'][pos])

    n = len(vecs)
    pairs = []
    total_constructive = 0
    total_destructive = 0
    for i in range(n):
        for j in range(i + 1, n):
            dot = float(jnp.dot(vecs[i], vecs[j]))
            cos = float(jnp.dot(vecs[i], vecs[j]) / (jnp.linalg.norm(vecs[i]) * jnp.linalg.norm(vecs[j]) + 1e-10))
            pairs.append({
                'component_a': names[i],
                'component_b': names[j],
                'dot_product': dot,
                'cosine': cos,
            })
            if dot > 0:
                total_constructive += dot
            else:
                total_destructive += dot

    return {
        'position': pos,
        'pairs': pairs,
        'total_constructive': total_constructive,
        'total_destructive': total_destructive,
    }
