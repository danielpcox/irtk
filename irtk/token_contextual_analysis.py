"""Token contextual analysis: how token representations change based on context."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def token_context_sensitivity(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """How much does context change a token's representation vs its embedding?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = tokens.shape[0] + position

    embed = cache['hook_embed'][position] + cache['hook_pos_embed'][position]
    embed_dir = embed / (jnp.linalg.norm(embed) + 1e-10)

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        resid_dir = resid / (jnp.linalg.norm(resid) + 1e-10)
        cos = float(jnp.dot(embed_dir, resid_dir))
        norm_ratio = float(jnp.linalg.norm(resid) / (jnp.linalg.norm(embed) + 1e-10))

        per_layer.append({
            'layer': layer,
            'cosine_to_embed': cos,
            'norm_ratio': norm_ratio,
            'context_influence': 1 - abs(cos),
        })

    return {
        'position': position,
        'token_id': int(tokens[position]),
        'embed_norm': float(jnp.linalg.norm(embed)),
        'per_layer': per_layer,
    }


def token_neighbor_influence(model: HookedTransformer, tokens: jnp.ndarray,
                              position: int = -1) -> dict:
    """How much does each neighboring token influence this position's representation?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]
    if position < 0:
        position = tokens.shape[0] + position

    # Aggregate attention to each source across all heads and layers
    source_weights = jnp.zeros(seq_len)
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        for head in range(n_heads):
            source_weights = source_weights + pattern[head, position, :]

    total = float(jnp.sum(source_weights))
    per_source = []
    for src in range(min(position + 1, seq_len)):
        per_source.append({
            'position': src,
            'token_id': int(tokens[src]),
            'total_attention': float(source_weights[src]),
            'fraction': float(source_weights[src] / (total + 1e-10)),
        })

    # Sort by attention
    per_source.sort(key=lambda x: x['total_attention'], reverse=True)

    return {
        'query_position': position,
        'per_source': per_source,
    }


def token_representation_divergence(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much do token representations diverge from each other through layers?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
        norms = jnp.linalg.norm(resid, axis=1, keepdims=True) + 1e-10
        normed = resid / norms
        cos_matrix = normed @ normed.T
        mask = 1 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(cos_matrix * mask) / (seq_len * (seq_len - 1) + 1e-10))

        per_layer.append({
            'layer': layer,
            'mean_pairwise_cosine': mean_sim,
            'diversity': 1 - mean_sim,
        })

    return {
        'per_layer': per_layer,
    }


def token_contextual_embedding(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int = -1) -> dict:
    """Contextual embedding analysis: how each token's representation relates to its embedding."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if layer < 0:
        layer = n_layers + layer
    seq_len = tokens.shape[0]

    per_token = []
    for pos in range(seq_len):
        embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
        resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
        embed_norm = float(jnp.linalg.norm(embed))
        resid_norm = float(jnp.linalg.norm(resid))

        cos = float(jnp.dot(embed, resid) / (embed_norm * resid_norm + 1e-10))

        per_token.append({
            'position': pos,
            'token_id': int(tokens[pos]),
            'embed_norm': embed_norm,
            'contextual_norm': resid_norm,
            'cosine_to_embed': cos,
            'context_shift': 1 - abs(cos),
        })

    return {
        'layer': layer,
        'per_token': per_token,
    }


def token_unique_information(model: HookedTransformer, tokens: jnp.ndarray,
                              position: int = -1) -> dict:
    """How much unique information does a token position carry?

    Measured by distance to nearest neighbor at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    if position < 0:
        position = tokens.shape[0] + position

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
        norms = jnp.linalg.norm(resid, axis=1, keepdims=True) + 1e-10
        normed = resid / norms

        sims = normed @ normed[position]  # [seq]
        sims = sims.at[position].set(-2.0)  # exclude self
        nearest_sim = float(jnp.max(sims))
        nearest_pos = int(jnp.argmax(sims))

        per_layer.append({
            'layer': layer,
            'nearest_cosine': nearest_sim,
            'nearest_position': nearest_pos,
            'uniqueness': 1 - nearest_sim,
        })

    return {
        'position': position,
        'per_layer': per_layer,
    }
