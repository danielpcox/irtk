"""Attention entropy dynamics: how attention entropy evolves across layers and positions."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def entropy_by_layer(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Per-layer average attention entropy across all heads."""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        # Entropy per head per position
        layer_entropies = []
        for head in range(n_heads):
            P = pattern[head]  # [seq, seq]
            ent = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)  # [seq]
            layer_entropies.append(float(jnp.mean(ent)))

        mean_ent = sum(layer_entropies) / len(layer_entropies)
        max_possible = float(jnp.mean(jnp.log(jnp.arange(1, seq_len + 1).astype(jnp.float32))))

        per_layer.append({
            'layer': layer,
            'mean_entropy': mean_ent,
            'normalized_entropy': mean_ent / (max_possible + 1e-10),
            'min_head_entropy': min(layer_entropies),
            'max_head_entropy': max(layer_entropies),
        })

    return {
        'per_layer': per_layer,
    }


def entropy_by_position(model: HookedTransformer, tokens: jnp.ndarray,
                        layer: int = 0) -> dict:
    """Attention entropy at each query position (averaged over heads)."""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]

    per_position = []
    for pos in range(seq_len):
        entropies = []
        for head in range(n_heads):
            row = pattern[head, pos, :pos + 1]
            ent = float(-jnp.sum(row * jnp.log(row + 1e-10)))
            entropies.append(ent)

        max_ent = float(jnp.log(pos + 1))
        mean_ent = sum(entropies) / len(entropies)

        per_position.append({
            'position': pos,
            'mean_entropy': mean_ent,
            'max_possible': max_ent,
            'normalized_entropy': mean_ent / (max_ent + 1e-10),
            'min_head': min(entropies),
            'max_head': max(entropies),
        })

    return {
        'layer': layer,
        'per_position': per_position,
    }


def entropy_head_evolution(model: HookedTransformer, tokens: jnp.ndarray,
                           head: int = 0) -> dict:
    """How does a specific head's entropy change across layers?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        P = pattern[head]  # [seq, seq]
        ent = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)  # [seq]
        mean_ent = float(jnp.mean(ent))
        max_prob = float(jnp.mean(jnp.max(P, axis=1)))

        per_layer.append({
            'layer': layer,
            'mean_entropy': mean_ent,
            'mean_max_prob': max_prob,
            'is_sharp': max_prob > 0.5,
        })

    return {
        'head': head,
        'per_layer': per_layer,
    }


def entropy_sharpening_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Does attention get sharper (lower entropy) in deeper layers?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    layer_entropies = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']
        ents = []
        for head in range(n_heads):
            P = pattern[head]
            ent = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)
            ents.append(float(jnp.mean(ent)))
        layer_entropies.append(sum(ents) / len(ents))

    # Trend: is later > earlier or vice versa?
    if len(layer_entropies) >= 2:
        first_half = sum(layer_entropies[:n_layers // 2]) / max(n_layers // 2, 1)
        second_half = sum(layer_entropies[n_layers // 2:]) / max(n_layers - n_layers // 2, 1)
        sharpening = first_half > second_half
    else:
        first_half = second_half = layer_entropies[0] if layer_entropies else 0
        sharpening = False

    return {
        'layer_entropies': layer_entropies,
        'first_half_mean': first_half,
        'second_half_mean': second_half,
        'is_sharpening': sharpening,
    }


def entropy_diversity_across_heads(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How diverse are entropy levels across heads within each layer?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_layer = []
    for layer in range(n_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']
        head_entropies = []
        for head in range(n_heads):
            P = pattern[head]
            ent = -jnp.sum(P * jnp.log(P + 1e-10), axis=1)
            head_entropies.append(float(jnp.mean(ent)))

        mean_ent = sum(head_entropies) / len(head_entropies)
        std_ent = (sum((e - mean_ent) ** 2 for e in head_entropies) / len(head_entropies)) ** 0.5

        per_layer.append({
            'layer': layer,
            'mean_entropy': mean_ent,
            'std_entropy': std_ent,
            'entropy_range': max(head_entropies) - min(head_entropies),
            'is_diverse': std_ent > 0.3,
        })

    return {
        'per_layer': per_layer,
    }
