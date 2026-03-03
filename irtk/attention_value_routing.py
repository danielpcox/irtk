"""Attention value routing: analyze how attention routes value information."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def value_source_routing(model: HookedTransformer, tokens: jnp.ndarray,
                          layer: int, head: int, position: int = -1) -> dict:
    """Where does value information come from for a specific query?"""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = tokens.shape[0] + position

    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]

    P = pattern[head, position, :position + 1]  # [pos+1]
    V = v[:position + 1, head, :]  # [pos+1, d_head]
    v_norms = jnp.linalg.norm(V, axis=1)  # [pos+1]

    per_source = []
    for src in range(position + 1):
        per_source.append({
            'position': src,
            'token_id': int(tokens[src]),
            'attention_weight': float(P[src]),
            'value_norm': float(v_norms[src]),
            'weighted_contribution': float(P[src] * v_norms[src]),
        })

    per_source.sort(key=lambda x: x['weighted_contribution'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'query_position': position,
        'per_source': per_source,
    }


def value_output_decomposition(model: HookedTransformer, tokens: jnp.ndarray,
                                layer: int, head: int, position: int = -1) -> dict:
    """Decompose head output into per-source contributions through OV circuit."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = tokens.shape[0] + position

    pattern = cache[f'blocks.{layer}.attn.hook_pattern']
    v = cache[f'blocks.{layer}.attn.hook_v']
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    P = pattern[head, position, :position + 1]
    V = v[:position + 1, head, :]

    # Total output
    total_out = jnp.sum(P[:, None] * V, axis=0) @ W_O[head]  # [d_model]
    total_norm = float(jnp.linalg.norm(total_out))

    per_source = []
    for src in range(position + 1):
        src_out = float(P[src]) * V[src] @ W_O[head]  # [d_model]
        src_norm = float(jnp.linalg.norm(src_out))
        # Alignment with total
        cos = float(jnp.dot(src_out, total_out) / (src_norm * total_norm + 1e-10))

        per_source.append({
            'position': src,
            'output_norm': src_norm,
            'fraction_of_total': src_norm / (total_norm + 1e-10),
            'alignment_with_total': cos,
        })

    per_source.sort(key=lambda x: x['output_norm'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'query_position': position,
        'total_output_norm': total_norm,
        'per_source': per_source,
    }


def value_routing_diversity(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int) -> dict:
    """How diverse is value routing across heads?"""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    pattern = cache[f'blocks.{layer}.attn.hook_pattern']
    v = cache[f'blocks.{layer}.attn.hook_v']
    W_O = model.blocks[layer].attn.W_O

    # Compute output direction for each head at last position
    dirs = []
    for head in range(n_heads):
        P = pattern[head, -1, :]
        V = v[:, head, :]
        out = jnp.sum(P[:, None] * V, axis=0) @ W_O[head]
        dirs.append(out / (jnp.linalg.norm(out) + 1e-10))

    dirs = jnp.stack(dirs)
    cos_matrix = dirs @ dirs.T

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                'head_a': i,
                'head_b': j,
                'output_cosine': float(cos_matrix[i, j]),
            })

    mask = 1 - jnp.eye(n_heads)
    mean_sim = float(jnp.sum(cos_matrix * mask) / (n_heads * (n_heads - 1) + 1e-10))

    return {
        'layer': layer,
        'mean_output_similarity': mean_sim,
        'is_diverse': mean_sim < 0.3,
        'pairs': pairs,
    }


def value_logit_routing(model: HookedTransformer, tokens: jnp.ndarray,
                         layer: int, head: int, position: int = -1) -> dict:
    """What vocabulary tokens does the routed value promote?"""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = tokens.shape[0] + position

    pattern = cache[f'blocks.{layer}.attn.hook_pattern']
    v = cache[f'blocks.{layer}.attn.hook_v']
    W_O = model.blocks[layer].attn.W_O
    W_U = model.unembed.W_U

    P = pattern[head, position, :position + 1]
    V = v[:position + 1, head, :]
    out = jnp.sum(P[:, None] * V, axis=0) @ W_O[head]  # [d_model]
    logits = out @ W_U  # [d_vocab]

    top_token = int(jnp.argmax(logits))
    bottom_token = int(jnp.argmin(logits))

    return {
        'layer': layer,
        'head': head,
        'position': position,
        'top_promoted': top_token,
        'top_promoted_logit': float(logits[top_token]),
        'top_suppressed': bottom_token,
        'top_suppressed_logit': float(logits[bottom_token]),
        'logit_range': float(logits[top_token] - logits[bottom_token]),
    }


def value_routing_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer summary of value routing patterns."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_layer = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']
        v = cache[f'blocks.{layer}.attn.hook_v']
        W_O = model.blocks[layer].attn.W_O

        head_norms = []
        for head in range(n_heads):
            P = pattern[head, -1, :]
            V = v[:, head, :]
            out = jnp.sum(P[:, None] * V, axis=0) @ W_O[head]
            head_norms.append(float(jnp.linalg.norm(out)))

        per_layer.append({
            'layer': layer,
            'mean_output_norm': sum(head_norms) / len(head_norms),
            'max_output_norm': max(head_norms),
            'dominant_head': head_norms.index(max(head_norms)),
        })

    return {
        'per_layer': per_layer,
    }
