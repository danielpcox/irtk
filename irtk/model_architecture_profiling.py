"""Model architecture profiling: understand the model's overall computational structure."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def parameter_count_profile(model: HookedTransformer) -> dict:
    """Count parameters by component type.

    Shows where the model's capacity is allocated.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_mlp = model.cfg.d_mlp

    embed_params = d_vocab * d_model  # W_E
    pos_params = model.cfg.n_ctx * d_model  # W_pos
    unembed_params = d_model * d_vocab + d_vocab  # W_U + b_U

    per_layer_attn = n_heads * d_model * d_head * 4  # Q, K, V, O
    per_layer_attn += n_heads * d_head * 2  # b_Q, b_K (if present)
    per_layer_mlp = d_model * d_mlp + d_mlp + d_mlp * d_model + d_model  # W_in, b_in, W_out, b_out
    per_layer_ln = d_model * 4  # 2 layer norms * (w + b)

    per_layer_total = per_layer_attn + per_layer_mlp + per_layer_ln
    total_layer_params = per_layer_total * n_layers

    total = embed_params + pos_params + unembed_params + total_layer_params

    return {
        'total_parameters': total,
        'embedding_params': embed_params,
        'positional_params': pos_params,
        'unembed_params': unembed_params,
        'per_layer_attn': per_layer_attn,
        'per_layer_mlp': per_layer_mlp,
        'per_layer_ln': per_layer_ln,
        'per_layer_total': per_layer_total,
        'total_layer_params': total_layer_params,
        'n_layers': n_layers,
        'embed_fraction': embed_params / total,
        'layer_fraction': total_layer_params / total,
    }


def computation_flow_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile the magnitude of computation at each stage.

    Shows where the model does most of its work.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed = cache['hook_embed']
    pos = cache['hook_pos_embed']
    embed_norm = float(jnp.mean(jnp.linalg.norm(embed + pos, axis=-1)))

    per_layer = []
    for layer in range(n_layers):
        attn_norm = float(jnp.mean(jnp.linalg.norm(
            cache[f'blocks.{layer}.hook_attn_out'], axis=-1
        )))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(
            cache[f'blocks.{layer}.hook_mlp_out'], axis=-1
        )))
        resid_norm = float(jnp.mean(jnp.linalg.norm(
            cache[f'blocks.{layer}.hook_resid_post'], axis=-1
        )))

        per_layer.append({
            'layer': layer,
            'attn_output_norm': attn_norm,
            'mlp_output_norm': mlp_norm,
            'residual_norm': resid_norm,
            'attn_fraction': attn_norm / (attn_norm + mlp_norm + 1e-10),
        })

    return {
        'embed_norm': embed_norm,
        'per_layer': per_layer,
    }


def hook_point_inventory(model: HookedTransformer) -> dict:
    """List all available hook points in the model.

    Useful for understanding what can be cached or intervened on.
    """
    n_layers = model.cfg.n_layers

    hook_points = []

    # Embedding hooks
    hook_points.append({'name': 'hook_embed', 'type': 'embed'})
    hook_points.append({'name': 'hook_pos_embed', 'type': 'embed'})

    for layer in range(n_layers):
        prefix = f'blocks.{layer}'
        # Residual stream hooks
        hook_points.append({'name': f'{prefix}.hook_resid_pre', 'type': 'residual'})
        hook_points.append({'name': f'{prefix}.hook_resid_mid', 'type': 'residual'})
        hook_points.append({'name': f'{prefix}.hook_resid_post', 'type': 'residual'})

        # Attention hooks
        hook_points.append({'name': f'{prefix}.hook_attn_out', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_q', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_k', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_v', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_z', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_attn_scores', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_pattern', 'type': 'attention'})
        hook_points.append({'name': f'{prefix}.attn.hook_result', 'type': 'attention'})

        # MLP hooks
        hook_points.append({'name': f'{prefix}.hook_mlp_out', 'type': 'mlp'})
        hook_points.append({'name': f'{prefix}.mlp.hook_pre', 'type': 'mlp'})
        hook_points.append({'name': f'{prefix}.mlp.hook_post', 'type': 'mlp'})

    return {
        'hook_points': hook_points,
        'total_hooks': len(hook_points),
        'n_embed': sum(1 for h in hook_points if h['type'] == 'embed'),
        'n_residual': sum(1 for h in hook_points if h['type'] == 'residual'),
        'n_attention': sum(1 for h in hook_points if h['type'] == 'attention'),
        'n_mlp': sum(1 for h in hook_points if h['type'] == 'mlp'),
    }


def model_capacity_utilization(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much of the model's capacity is being used?

    Measures effective rank and utilization of weight matrices.
    """
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        # OV matrix effective rank
        W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
        W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

        # Stack OV circuits
        n_heads = W_V.shape[0]
        ov_ranks = []
        for h in range(n_heads):
            OV = W_V[h] @ W_O[h]  # [d_model, d_model]
            S = jnp.linalg.svd(OV, compute_uv=False)
            S_norm = S / (jnp.sum(S) + 1e-10)
            ent = -float(jnp.sum(S_norm * jnp.log(S_norm + 1e-10)))
            ov_ranks.append(float(jnp.exp(ent)))

        # MLP weight rank
        W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
        S_mlp = jnp.linalg.svd(W_in, compute_uv=False)
        S_mlp_norm = S_mlp / (jnp.sum(S_mlp) + 1e-10)
        mlp_ent = -float(jnp.sum(S_mlp_norm * jnp.log(S_mlp_norm + 1e-10)))
        mlp_rank = float(jnp.exp(mlp_ent))

        per_layer.append({
            'layer': layer,
            'mean_ov_rank': sum(ov_ranks) / len(ov_ranks),
            'mlp_effective_rank': mlp_rank,
        })

    return {
        'per_layer': per_layer,
    }


def model_summary(model: HookedTransformer) -> dict:
    """Generate a concise model summary.

    Architecture details at a glance.
    """
    cfg = model.cfg
    return {
        'n_layers': cfg.n_layers,
        'd_model': cfg.d_model,
        'n_heads': cfg.n_heads,
        'd_head': cfg.d_head,
        'd_mlp': cfg.d_mlp,
        'd_vocab': cfg.d_vocab,
        'n_ctx': cfg.n_ctx,
        'act_fn': cfg.act_fn,
        'normalization_type': cfg.normalization_type,
        'positional_embedding_type': cfg.positional_embedding_type,
    }
