"""Head writing analysis: analyze what attention heads write to the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def head_writing_directions(model: HookedTransformer, tokens: jnp.ndarray,
                             layer: int) -> dict:
    """What direction does each head write into the residual stream?"""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for head in range(n_heads):
        # Head output: z @ W_O for last position
        head_out = z[-1, head] @ W_O[head]  # [d_model]
        out_norm = float(jnp.linalg.norm(head_out))
        out_dir = head_out / (out_norm + 1e-10)

        # Mean output direction across positions
        mean_out = jnp.mean(z[:, head] @ W_O[head], axis=0)
        mean_norm = float(jnp.linalg.norm(mean_out))

        per_head.append({
            'head': head,
            'output_norm': out_norm,
            'mean_output_norm': mean_norm,
        })

    # Cross-head direction similarity
    dirs = []
    for head in range(n_heads):
        head_out = z[-1, head] @ W_O[head]
        dirs.append(head_out / (jnp.linalg.norm(head_out) + 1e-10))
    dirs = jnp.stack(dirs)
    cos_matrix = dirs @ dirs.T

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                'head_a': i,
                'head_b': j,
                'cosine': float(cos_matrix[i, j]),
            })

    return {
        'layer': layer,
        'per_head': per_head,
        'direction_pairs': pairs,
    }


def head_logit_writing(model: HookedTransformer, tokens: jnp.ndarray,
                        layer: int, position: int = -1) -> dict:
    """What logit effect does each head's output have?"""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    if position < 0:
        position = tokens.shape[0] + position

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_head = []
    for head in range(n_heads):
        head_out = z[position, head] @ W_O[head]  # [d_model]
        logits = head_out @ W_U  # [d_vocab]
        top_token = int(jnp.argmax(logits))
        bottom_token = int(jnp.argmin(logits))

        per_head.append({
            'head': head,
            'top_promoted_token': top_token,
            'top_promoted_logit': float(logits[top_token]),
            'top_suppressed_token': bottom_token,
            'top_suppressed_logit': float(logits[bottom_token]),
            'logit_range': float(logits[top_token] - logits[bottom_token]),
        })

    return {
        'layer': layer,
        'position': position,
        'per_head': per_head,
    }


def head_writing_consistency(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int) -> dict:
    """Is each head's writing direction consistent across positions?"""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for head in range(n_heads):
        outputs = z[:, head] @ W_O[head]  # [seq, d_model]
        norms = jnp.linalg.norm(outputs, axis=1, keepdims=True) + 1e-10
        normed = outputs / norms

        # Mean pairwise cosine
        cos_matrix = normed @ normed.T
        mask = 1 - jnp.eye(seq_len)
        mean_cos = float(jnp.sum(cos_matrix * mask) / (seq_len * (seq_len - 1) + 1e-10))

        per_head.append({
            'head': head,
            'mean_direction_consistency': mean_cos,
            'is_consistent': mean_cos > 0.5,
            'mean_output_norm': float(jnp.mean(norms)),
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }


def head_residual_contribution(model: HookedTransformer, tokens: jnp.ndarray,
                                position: int = -1) -> dict:
    """How much does each head contribute to the residual stream norm?"""
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = tokens.shape[0] + position

    per_head = []
    for layer in range(n_layers):
        z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
        W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
        resid = cache[f'blocks.{layer}.hook_resid_post'][position]
        resid_dir = resid / (jnp.linalg.norm(resid) + 1e-10)

        for head in range(n_heads):
            head_out = z[position, head] @ W_O[head]
            contrib_norm = float(jnp.linalg.norm(head_out))
            alignment = float(jnp.dot(head_out, resid_dir))

            per_head.append({
                'layer': layer,
                'head': head,
                'output_norm': contrib_norm,
                'residual_alignment': alignment,
                'is_constructive': alignment > 0,
            })

    return {
        'position': position,
        'per_head': per_head,
    }


def head_writing_rank(model: HookedTransformer, tokens: jnp.ndarray,
                       layer: int) -> dict:
    """Effective rank of each head's output across positions."""
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for head in range(n_heads):
        outputs = z[:, head] @ W_O[head]  # [seq, d_model]
        sv = jnp.linalg.svd(outputs, compute_uv=False)
        sv_norm = sv / (jnp.sum(sv) + 1e-10)
        entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
        eff_rank = float(jnp.exp(entropy))

        per_head.append({
            'head': head,
            'effective_rank': eff_rank,
            'top_sv_fraction': float(sv[0] / (jnp.sum(sv) + 1e-10)),
            'is_low_rank': eff_rank < 2.0,
        })

    return {
        'layer': layer,
        'per_head': per_head,
    }
