"""Residual stream statistics: detailed statistical analysis of the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def residual_norm_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track the norm of the residual stream through layers.

    Shows how norm grows, which layers contribute most.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed = cache['hook_embed'] + cache['hook_pos_embed']
    embed_norm = float(jnp.mean(jnp.linalg.norm(embed, axis=-1)))

    per_layer = []
    prev_norm = embed_norm
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))
        growth = norm - prev_norm

        per_layer.append({
            'layer': layer,
            'mean_norm': norm,
            'norm_growth': growth,
            'growth_rate': growth / (prev_norm + 1e-10),
        })
        prev_norm = norm

    total_growth = per_layer[-1]['mean_norm'] - embed_norm if per_layer else 0

    return {
        'embed_norm': embed_norm,
        'per_layer': per_layer,
        'total_growth': total_growth,
        'final_norm': per_layer[-1]['mean_norm'] if per_layer else embed_norm,
    }


def residual_direction_drift(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much does the residual direction change across layers?

    Cosine similarity between consecutive layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed = cache['hook_embed'] + cache['hook_pos_embed']
    prev = embed

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        # Mean cosine similarity across positions
        cos = float(jnp.mean(
            jnp.sum(resid * prev, axis=-1) /
            (jnp.linalg.norm(resid, axis=-1) * jnp.linalg.norm(prev, axis=-1) + 1e-10)
        ))

        # Also track similarity to embedding
        cos_to_embed = float(jnp.mean(
            jnp.sum(resid * embed, axis=-1) /
            (jnp.linalg.norm(resid, axis=-1) * jnp.linalg.norm(embed, axis=-1) + 1e-10)
        ))

        per_layer.append({
            'layer': layer,
            'cos_to_previous': cos,
            'cos_to_embed': cos_to_embed,
        })
        prev = resid

    return {
        'per_layer': per_layer,
        'mean_drift': 1.0 - sum(p['cos_to_previous'] for p in per_layer) / len(per_layer) if per_layer else 0,
    }


def residual_variance_decomposition(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Decompose the variance of the residual stream at each layer.

    How much variance comes from embedding, attention, MLP?
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
        attn_out = cache[f'blocks.{layer}.hook_attn_out']
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']

        resid_var = float(jnp.var(resid))
        attn_var = float(jnp.var(attn_out))
        mlp_var = float(jnp.var(mlp_out))
        total_comp_var = attn_var + mlp_var

        per_layer.append({
            'layer': layer,
            'residual_variance': resid_var,
            'attention_variance': attn_var,
            'mlp_variance': mlp_var,
            'attn_fraction': attn_var / (total_comp_var + 1e-10),
            'mlp_fraction': mlp_var / (total_comp_var + 1e-10),
        })

    return {
        'per_layer': per_layer,
    }


def residual_position_similarity(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How similar are the residual vectors at different positions?

    High similarity = positions converge; low = diverse representations.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
    seq_len = tokens.shape[0]

    normed = resid / (jnp.linalg.norm(resid, axis=-1, keepdims=True) + 1e-10)
    sim_matrix = normed @ normed.T  # [seq, seq]

    # Extract upper triangle (exclude diagonal)
    pairs = []
    total_sim = 0.0
    n_pairs = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            s = float(sim_matrix[i, j])
            pairs.append({'pos_a': i, 'pos_b': j, 'similarity': s})
            total_sim += s
            n_pairs += 1

    mean_sim = total_sim / n_pairs if n_pairs > 0 else 0.0

    return {
        'layer': layer,
        'pairs': sorted(pairs, key=lambda x: x['similarity'], reverse=True),
        'mean_similarity': mean_sim,
        'is_converged': mean_sim > 0.8,
    }


def residual_component_balance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How balanced are attention and MLP contributions at each layer?

    Measures relative magnitude and alignment.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']  # [seq, d_model]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']

        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=-1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))

        # Alignment between attn and mlp
        cos = float(jnp.mean(
            jnp.sum(attn_out * mlp_out, axis=-1) /
            (jnp.linalg.norm(attn_out, axis=-1) * jnp.linalg.norm(mlp_out, axis=-1) + 1e-10)
        ))

        ratio = attn_norm / (mlp_norm + 1e-10)
        dominant = 'attn' if ratio > 1.0 else 'mlp'

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'attn_mlp_ratio': ratio,
            'alignment': cos,
            'dominant': dominant,
            'is_cooperative': cos > 0,
        })

    return {
        'per_layer': per_layer,
        'n_attn_dominant': sum(1 for p in per_layer if p['dominant'] == 'attn'),
        'n_mlp_dominant': sum(1 for p in per_layer if p['dominant'] == 'mlp'),
    }
