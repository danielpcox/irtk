"""Attention head output analysis: analyze what each head writes to the residual stream."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def head_writing_direction(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """What direction does each head write in?

    Computes the mean output direction and consistency across positions.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])  # [seq, d_model]
        mean_out = jnp.mean(head_out, axis=0)
        mean_dir = mean_out / (jnp.linalg.norm(mean_out) + 1e-10)

        # Consistency: how aligned is each position's output with the mean direction?
        normed = head_out / (jnp.linalg.norm(head_out, axis=-1, keepdims=True) + 1e-10)
        consistency = float(jnp.mean(normed @ mean_dir))

        per_head.append({
            'head': h,
            'mean_output_norm': float(jnp.linalg.norm(mean_out)),
            'direction_consistency': consistency,
            'is_consistent': consistency > 0.5,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'n_consistent': sum(1 for h in per_head if h['is_consistent']),
    }


def head_unembed_alignment(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """How aligned is each head's output with unembedding directions?

    High alignment = head directly promotes specific tokens.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('h,hm->m', z[pos, h, :], W_O[h])  # [d_model]
        head_dir = head_out / (jnp.linalg.norm(head_out) + 1e-10)

        # Compute alignment with all unembedding directions
        W_U_normed = W_U / (jnp.linalg.norm(W_U, axis=0, keepdims=True) + 1e-10)
        alignments = head_dir @ W_U_normed  # [d_vocab]

        max_align_token = int(jnp.argmax(alignments))
        max_alignment = float(alignments[max_align_token])
        mean_abs_alignment = float(jnp.mean(jnp.abs(alignments)))

        per_head.append({
            'head': h,
            'max_alignment': max_alignment,
            'max_aligned_token': max_align_token,
            'mean_abs_alignment': mean_abs_alignment,
            'is_token_specific': max_alignment > 0.3,
        })

    return {
        'layer': layer,
        'position': pos,
        'per_head': per_head,
    }


def head_output_diversity(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How diverse are the outputs across heads?

    Measures pairwise similarity between head output directions.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']
    W_O = model.blocks[layer].attn.W_O

    # Get mean output direction per head
    mean_dirs = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])
        mean_out = jnp.mean(head_out, axis=0)
        mean_dir = mean_out / (jnp.linalg.norm(mean_out) + 1e-10)
        mean_dirs.append(mean_dir)

    mean_dirs = jnp.stack(mean_dirs)  # [n_heads, d_model]
    sim_matrix = mean_dirs @ mean_dirs.T  # [n_heads, n_heads]

    pairs = []
    total_sim = 0.0
    n_pairs = 0
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            s = float(sim_matrix[i, j])
            pairs.append({'head_a': i, 'head_b': j, 'similarity': s})
            total_sim += abs(s)
            n_pairs += 1

    mean_abs_sim = total_sim / n_pairs if n_pairs > 0 else 0.0

    return {
        'layer': layer,
        'pairs': pairs,
        'mean_abs_similarity': mean_abs_sim,
        'is_diverse': mean_abs_sim < 0.5,
    }


def head_position_specialization(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Does each head specialize on certain positions?

    Measures variance of output norms across positions.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    z = cache[f'blocks.{layer}.attn.hook_z']
    W_O = model.blocks[layer].attn.W_O

    per_head = []
    for h in range(n_heads):
        head_out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])
        norms = jnp.linalg.norm(head_out, axis=-1)  # [seq]
        mean_norm = float(jnp.mean(norms))
        std_norm = float(jnp.std(norms))
        max_pos = int(jnp.argmax(norms))
        cv = std_norm / (mean_norm + 1e-10)

        per_head.append({
            'head': h,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'coefficient_of_variation': cv,
            'max_position': max_pos,
            'is_position_specialized': cv > 0.5,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'n_specialized': sum(1 for h in per_head if h['is_position_specialized']),
    }


def head_combined_effect(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Analyze the combined effect of all heads at a position.

    Decompose the total attention output into constructive/destructive parts.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_heads = model.cfg.n_heads

    z = cache[f'blocks.{layer}.attn.hook_z']
    W_O = model.blocks[layer].attn.W_O

    head_outs = []
    for h in range(n_heads):
        out = jnp.einsum('h,hm->m', z[pos, h, :], W_O[h])
        head_outs.append(out)

    head_outs = jnp.stack(head_outs)  # [n_heads, d_model]
    combined = jnp.sum(head_outs, axis=0)
    combined_dir = combined / (jnp.linalg.norm(combined) + 1e-10)

    per_head = []
    for h in range(n_heads):
        # Project onto combined direction
        projection = float(jnp.dot(head_outs[h], combined_dir))
        norm = float(jnp.linalg.norm(head_outs[h]))

        per_head.append({
            'head': h,
            'norm': norm,
            'projection_onto_combined': projection,
            'is_constructive': projection > 0,
            'fraction_constructive': max(0, projection) / (norm + 1e-10),
        })

    total_constructive = sum(max(0, h['projection_onto_combined']) for h in per_head)
    total_destructive = sum(max(0, -h['projection_onto_combined']) for h in per_head)

    return {
        'layer': layer,
        'position': pos,
        'combined_norm': float(jnp.linalg.norm(combined)),
        'per_head': per_head,
        'total_constructive': total_constructive,
        'total_destructive': total_destructive,
        'efficiency': float(jnp.linalg.norm(combined)) / (sum(h['norm'] for h in per_head) + 1e-10),
    }
