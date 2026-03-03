"""Logit circuit tracing: trace circuits from input to specific logits."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def trace_logit_to_components(model: HookedTransformer, tokens: jnp.ndarray, target_token: int, position: int = -1) -> dict:
    """Trace a target token's logit back to each component's contribution.

    Decomposes the logit into: embedding + sum of attention + MLP outputs.
    """
    logits, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    unembed_dir = W_U[:, target_token]  # [d_model]

    components = []

    # Embedding contribution
    embed_key = 'hook_embed'
    if embed_key in cache:
        embed = cache[embed_key][pos]
        pos_key = 'hook_pos_embed'
        if pos_key in cache:
            embed = embed + cache[pos_key][pos]
        embed_logit = float(jnp.dot(embed, unembed_dir))
        components.append({
            'name': 'embed',
            'layer': -1,
            'type': 'embedding',
            'logit_contribution': embed_logit,
        })

    # Per-layer attn and MLP contributions
    for layer in range(n_layers):
        attn_key = f'blocks.{layer}.hook_attn_out'
        mlp_key = f'blocks.{layer}.hook_mlp_out'

        attn_out = cache[attn_key][pos]
        mlp_out = cache[mlp_key][pos]

        attn_logit = float(jnp.dot(attn_out, unembed_dir))
        mlp_logit = float(jnp.dot(mlp_out, unembed_dir))

        components.append({
            'name': f'L{layer}_attn',
            'layer': layer,
            'type': 'attention',
            'logit_contribution': attn_logit,
        })
        components.append({
            'name': f'L{layer}_mlp',
            'layer': layer,
            'type': 'mlp',
            'logit_contribution': mlp_logit,
        })

    # Bias contribution
    if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None:
        bias_logit = float(model.unembed.b_U[target_token])
        components.append({
            'name': 'bias',
            'layer': -1,
            'type': 'bias',
            'logit_contribution': bias_logit,
        })

    total = sum(c['logit_contribution'] for c in components)
    actual_logit = float(logits[pos, target_token])

    components.sort(key=lambda x: abs(x['logit_contribution']), reverse=True)

    return {
        'target_token': target_token,
        'position': pos,
        'components': components,
        'total_traced': total,
        'actual_logit': actual_logit,
    }


def per_head_logit_contribution(model: HookedTransformer, tokens: jnp.ndarray, target_token: int, layer: int, position: int = -1) -> dict:
    """Break down a layer's attention logit contribution by head.

    Each head's output projected onto the target token's unembedding direction.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    W_U = model.unembed.W_U
    unembed_dir = W_U[:, target_token]

    z_key = f'blocks.{layer}.attn.hook_z'
    z = cache[z_key]
    W_O = model.blocks[layer].attn.W_O

    per_head = []
    total = 0.0
    for h in range(n_heads):
        head_out = jnp.einsum('h,hm->m', z[pos, h], W_O[h])
        logit = float(jnp.dot(head_out, unembed_dir))
        norm = float(jnp.linalg.norm(head_out))
        per_head.append({
            'head': h,
            'logit_contribution': logit,
            'output_norm': norm,
            'promotes': logit > 0,
        })
        total += logit

    per_head.sort(key=lambda x: abs(x['logit_contribution']), reverse=True)

    return {
        'layer': layer,
        'target_token': target_token,
        'position': pos,
        'per_head': per_head,
        'total_attn_logit': total,
    }


def logit_attribution_path(model: HookedTransformer, tokens: jnp.ndarray, target_token: int, position: int = -1) -> dict:
    """Trace the cumulative path of logit formation through layers.

    Shows how the target token's logit builds up layer by layer.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    unembed_dir = W_U[:, target_token]

    per_layer = []
    cumulative = 0.0

    # Start with embedding
    embed_key = 'hook_embed'
    if embed_key in cache:
        embed = cache[embed_key][pos]
        pos_key = 'hook_pos_embed'
        if pos_key in cache:
            embed = embed + cache[pos_key][pos]
        cumulative += float(jnp.dot(embed, unembed_dir))

    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]

        attn_contrib = float(jnp.dot(attn_out, unembed_dir))
        mlp_contrib = float(jnp.dot(mlp_out, unembed_dir))

        cumulative += attn_contrib + mlp_contrib

        per_layer.append({
            'layer': layer,
            'attn_contribution': attn_contrib,
            'mlp_contribution': mlp_contrib,
            'cumulative_logit': cumulative,
        })

    return {
        'target_token': target_token,
        'position': pos,
        'per_layer': per_layer,
        'final_logit': cumulative,
    }


def competing_logit_analysis(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1, top_k: int = 5) -> dict:
    """Analyze which tokens compete for the highest logit.

    For the top-k tokens, shows per-layer contribution comparison.
    """
    logits, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    top_tokens = jnp.argsort(logits[pos])[-top_k:][::-1]

    per_token = []
    for t_idx in top_tokens:
        t = int(t_idx)
        unembed_dir = W_U[:, t]
        total = 0.0
        per_layer = []

        for layer in range(n_layers):
            attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
            mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]
            layer_contrib = float(jnp.dot(attn_out + mlp_out, unembed_dir))
            total += layer_contrib
            per_layer.append({'layer': layer, 'contribution': layer_contrib})

        per_token.append({
            'token': t,
            'final_logit': float(logits[pos, t]),
            'per_layer': per_layer,
        })

    winner = per_token[0]['token']
    margin = per_token[0]['final_logit'] - per_token[1]['final_logit'] if len(per_token) > 1 else 0.0

    return {
        'position': pos,
        'per_token': per_token,
        'winner': winner,
        'margin': margin,
    }


def critical_circuit_components(model: HookedTransformer, tokens: jnp.ndarray, target_token: int, position: int = -1, threshold: float = 0.1) -> dict:
    """Identify which components are critical for a target token's logit.

    Components contributing more than threshold fraction are critical.
    """
    result = trace_logit_to_components(model, tokens, target_token, position)
    components = result['components']
    total_abs = sum(abs(c['logit_contribution']) for c in components)

    for c in components:
        c['fraction'] = abs(c['logit_contribution']) / (total_abs + 1e-10)
        c['is_critical'] = c['fraction'] > threshold

    critical = [c for c in components if c['is_critical']]
    critical_logit = sum(c['logit_contribution'] for c in critical)

    return {
        'target_token': target_token,
        'position': result['position'],
        'components': components,
        'critical_components': critical,
        'n_critical': len(critical),
        'critical_logit_fraction': critical_logit / (result['actual_logit'] + 1e-10) if result['actual_logit'] != 0 else 0.0,
    }
