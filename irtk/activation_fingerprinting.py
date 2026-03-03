"""Activation fingerprinting: characterize model behavior via activation signatures."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def layer_activation_fingerprint(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Create a compact fingerprint of each layer's activation behavior.

    Summarizes each layer by norm, direction, and entropy of the residual.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
        norms = jnp.linalg.norm(resid, axis=-1)  # [seq]
        mean_norm = float(jnp.mean(norms))
        std_norm = float(jnp.std(norms))

        # Direction: cosine similarity between first and last position
        cos_first_last = float(
            jnp.dot(resid[0], resid[-1]) /
            (jnp.linalg.norm(resid[0]) * jnp.linalg.norm(resid[-1]) + 1e-10)
        )

        # Mean pairwise cosine similarity
        normed = resid / (jnp.linalg.norm(resid, axis=-1, keepdims=True) + 1e-10)
        sim_matrix = normed @ normed.T
        mean_sim = float(jnp.mean(sim_matrix))

        per_layer.append({
            'layer': layer,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'cos_first_last': cos_first_last,
            'mean_pairwise_similarity': mean_sim,
        })

    return {
        'per_layer': per_layer,
        'n_layers': n_layers,
    }


def head_output_fingerprint(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Fingerprint each head's output at a given layer.

    Captures norm, direction, and rank contribution of each head.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])  # [seq, d_model]
        norms = jnp.linalg.norm(head_out, axis=-1)
        mean_norm = float(jnp.mean(norms))
        max_norm = float(jnp.max(norms))

        # Direction consistency across positions
        normed = head_out / (jnp.linalg.norm(head_out, axis=-1, keepdims=True) + 1e-10)
        mean_dir = jnp.mean(normed, axis=0)
        mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)
        alignment = float(jnp.mean(normed @ mean_dir))

        per_head.append({
            'head': h,
            'mean_norm': mean_norm,
            'max_norm': max_norm,
            'direction_consistency': alignment,
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def mlp_activation_fingerprint(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Fingerprint MLP activation patterns across layers.

    Captures sparsity, magnitude, and active neuron count.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        post_key = f'blocks.{layer}.mlp.hook_post'
        if post_key not in cache:
            continue
        post = cache[post_key]  # [seq, d_mlp]
        active = jnp.sum(post > 0, axis=-1)  # [seq]
        total_neurons = post.shape[-1]

        mean_active = float(jnp.mean(active))
        sparsity = 1.0 - mean_active / total_neurons
        mean_mag = float(jnp.mean(jnp.abs(post)))
        max_mag = float(jnp.max(jnp.abs(post)))

        per_layer.append({
            'layer': layer,
            'mean_active_neurons': mean_active,
            'sparsity': sparsity,
            'mean_magnitude': mean_mag,
            'max_magnitude': max_mag,
        })

    return {
        'per_layer': per_layer,
        'n_layers': len(per_layer),
    }


def attention_pattern_fingerprint(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Fingerprint attention patterns across all layers and heads.

    Captures entropy, sparsity, and dominant pattern type.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_layer = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        per_head = []

        for h in range(n_heads):
            p = patterns[h]  # [seq, seq]
            # Entropy per query position
            entropy = -jnp.sum(p * jnp.log(p + 1e-10), axis=-1)  # [seq]
            mean_entropy = float(jnp.mean(entropy))

            # Diagonal attention (self-attention)
            diag_mass = float(jnp.mean(jnp.diag(p)))

            # Previous token attention
            if seq_len > 1:
                prev_mass = float(jnp.mean(jnp.array([
                    float(p[i, i-1]) for i in range(1, seq_len)
                ])))
            else:
                prev_mass = 0.0

            # BOS attention
            bos_mass = float(jnp.mean(p[:, 0]))

            per_head.append({
                'head': h,
                'mean_entropy': mean_entropy,
                'self_attention': diag_mass,
                'prev_token_attention': prev_mass,
                'bos_attention': bos_mass,
            })

        per_layer.append({
            'layer': layer,
            'per_head': per_head,
        })

    return {
        'per_layer': per_layer,
    }


def input_sensitivity_fingerprint(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Fingerprint how sensitive each layer is to input token changes.

    Measures residual change per position relative to embedding.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed = cache['hook_embed'] + cache['hook_pos_embed']  # [seq, d_model]
    embed_norms = jnp.linalg.norm(embed, axis=-1)  # [seq]

    per_layer = []
    prev_resid = embed
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        delta = resid - prev_resid
        delta_norms = jnp.linalg.norm(delta, axis=-1)  # [seq]

        # Relative change
        rel_change = float(jnp.mean(delta_norms / (jnp.linalg.norm(prev_resid, axis=-1) + 1e-10)))

        # Direction change
        cos_sim = float(jnp.mean(
            jnp.sum(resid * prev_resid, axis=-1) /
            (jnp.linalg.norm(resid, axis=-1) * jnp.linalg.norm(prev_resid, axis=-1) + 1e-10)
        ))

        per_layer.append({
            'layer': layer,
            'mean_delta_norm': float(jnp.mean(delta_norms)),
            'relative_change': rel_change,
            'direction_preservation': cos_sim,
        })
        prev_resid = resid

    return {
        'per_layer': per_layer,
    }
