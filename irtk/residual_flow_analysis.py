"""Residual flow analysis: track how information flows through the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def residual_direction_flow(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """How does the residual stream direction evolve layer by layer?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    directions = []
    # Embedding
    resid = cache['hook_embed'][position] + cache['hook_pos_embed'][position]
    directions.append(resid / (jnp.linalg.norm(resid) + 1e-10))

    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        directions.append(resid / (jnp.linalg.norm(resid) + 1e-10))

    per_layer = []
    for i in range(len(directions) - 1):
        cos = float(jnp.dot(directions[i], directions[i + 1]))
        per_layer.append({
            'from_layer': i - 1 if i > 0 else 'embed',
            'to_layer': i,
            'direction_cosine': cos,
            'direction_change': 1 - abs(cos),
        })

    # Total drift
    total_drift = float(jnp.dot(directions[0], directions[-1]))

    return {
        'position': position,
        'per_layer': per_layer,
        'embed_to_final_cosine': total_drift,
    }


def residual_norm_flow(model: HookedTransformer, tokens: jnp.ndarray,
                       position: int = -1) -> dict:
    """Track norm growth/decay through the residual stream."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    embed = cache['hook_embed'][position] + cache['hook_pos_embed'][position]
    embed_norm = float(jnp.linalg.norm(embed))

    per_layer = []
    prev_norm = embed_norm
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        norm = float(jnp.linalg.norm(resid))
        growth = norm / (prev_norm + 1e-10)
        per_layer.append({
            'layer': layer,
            'norm': norm,
            'growth_factor': growth,
            'cumulative_growth': norm / (embed_norm + 1e-10),
        })
        prev_norm = norm

    return {
        'position': position,
        'embed_norm': embed_norm,
        'final_norm': per_layer[-1]['norm'] if per_layer else embed_norm,
        'per_layer': per_layer,
    }


def residual_component_flow(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """Decompose residual stream flow into attention vs MLP contributions."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][position]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][position]
        resid_post = cache[f'blocks.{layer}.hook_resid_post'][position]

        attn_norm = float(jnp.linalg.norm(attn_out))
        mlp_norm = float(jnp.linalg.norm(mlp_out))
        total = attn_norm + mlp_norm + 1e-10

        # Alignment between attn and mlp
        cos = float(jnp.dot(attn_out, mlp_out) /
                     (attn_norm * mlp_norm + 1e-10))

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'attn_fraction': attn_norm / total,
            'mlp_fraction': mlp_norm / total,
            'attn_mlp_cosine': cos,
            'resid_norm': float(jnp.linalg.norm(resid_post)),
        })

    return {
        'position': position,
        'per_layer': per_layer,
    }


def residual_signal_noise(model: HookedTransformer, tokens: jnp.ndarray,
                          position: int = -1) -> dict:
    """Decompose residual into signal (target direction) and noise."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    # Target direction: final residual normalized
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post'][position]
    target_dir = final_resid / (jnp.linalg.norm(final_resid) + 1e-10)

    embed = cache['hook_embed'][position] + cache['hook_pos_embed'][position]

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        signal = float(jnp.dot(resid, target_dir))
        noise = float(jnp.sqrt(jnp.maximum(jnp.dot(resid, resid) - signal ** 2, 0)))

        per_layer.append({
            'layer': layer,
            'signal': signal,
            'noise': noise,
            'snr': signal / (noise + 1e-10),
        })

    # Embedding signal
    embed_signal = float(jnp.dot(embed, target_dir))

    return {
        'position': position,
        'embed_signal': embed_signal,
        'per_layer': per_layer,
    }


def residual_cross_position_flow(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How similar is residual flow across different positions?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    # Collect final residuals
    final = cache[f'blocks.{n_layers - 1}.hook_resid_post']  # [seq, d_model]
    norms = jnp.linalg.norm(final, axis=1, keepdims=True) + 1e-10
    normed = final / norms

    # Pairwise cosines
    cos_matrix = normed @ normed.T

    per_position = []
    for pos in range(seq_len):
        mean_sim = float((jnp.sum(cos_matrix[pos]) - 1) / max(seq_len - 1, 1))
        per_position.append({
            'position': pos,
            'norm': float(norms[pos, 0]),
            'mean_similarity_to_others': mean_sim,
        })

    # Overall diversity
    mask = 1 - jnp.eye(seq_len)
    mean_pairwise = float(jnp.sum(cos_matrix * mask) / (seq_len * (seq_len - 1) + 1e-10))

    return {
        'per_position': per_position,
        'mean_pairwise_similarity': mean_pairwise,
        'is_diverse': mean_pairwise < 0.5,
    }
