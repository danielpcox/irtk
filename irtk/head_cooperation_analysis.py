"""Head cooperation analysis: how heads cooperate and compete."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def within_layer_cooperation(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Do heads within a layer cooperate or compete?

    Measures pairwise cosine similarity of head outputs at a given position.
    Positive = cooperative, negative = competing.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    z_key = f'blocks.{layer}.attn.hook_z'
    z = cache[z_key]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    n_heads = model.cfg.n_heads

    # Compute per-head outputs at position
    head_outputs = []
    for h in range(n_heads):
        out = jnp.einsum('h,hm->m', z[pos, h], W_O[h])
        head_outputs.append(out)
    head_outputs = jnp.stack(head_outputs)  # [n_heads, d_model]

    # Pairwise cosine similarity
    norms = jnp.linalg.norm(head_outputs, axis=-1, keepdims=True) + 1e-10
    normed = head_outputs / norms
    sim_matrix = normed @ normed.T  # [n_heads, n_heads]

    pairs = []
    total_sim = 0.0
    n_pairs = 0
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            sim = float(sim_matrix[i, j])
            pairs.append({
                'head_a': i,
                'head_b': j,
                'cosine_similarity': sim,
                'cooperates': sim > 0,
            })
            total_sim += sim
            n_pairs += 1

    mean_cooperation = total_sim / n_pairs if n_pairs > 0 else 0.0

    return {
        'layer': layer,
        'position': pos,
        'pairs': pairs,
        'mean_cooperation': mean_cooperation,
        'is_cooperative': mean_cooperation > 0,
        'n_cooperative_pairs': sum(1 for p in pairs if p['cooperates']),
        'n_competing_pairs': sum(1 for p in pairs if not p['cooperates']),
    }


def cross_layer_head_alignment(model: HookedTransformer, tokens: jnp.ndarray, head: int) -> dict:
    """Track alignment of the same head index across layers.

    Are head N's outputs aligned at different depths?
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]

    # Collect head outputs at last position
    outputs = []
    for layer in range(n_layers):
        z_key = f'blocks.{layer}.attn.hook_z'
        z = cache[z_key]
        W_O = model.blocks[layer].attn.W_O
        out = jnp.einsum('h,hm->m', z[-1, head], W_O[head])
        outputs.append(out)

    outputs = jnp.stack(outputs)  # [n_layers, d_model]

    # Pairwise alignment
    norms = jnp.linalg.norm(outputs, axis=-1, keepdims=True) + 1e-10
    normed = outputs / norms

    per_layer = []
    for layer in range(n_layers):
        norm = float(jnp.linalg.norm(outputs[layer]))
        if layer > 0:
            align = float(jnp.dot(normed[layer], normed[layer-1]))
        else:
            align = 1.0
        per_layer.append({
            'layer': layer,
            'output_norm': norm,
            'alignment_to_previous': align,
        })

    # Mean pairwise
    sim_matrix = normed @ normed.T
    mean_sim = float(jnp.mean(sim_matrix[jnp.triu_indices(n_layers, k=1)]))

    return {
        'head': head,
        'per_layer': per_layer,
        'mean_alignment': mean_sim,
        'is_consistent': mean_sim > 0.3,
    }


def head_redundancy_analysis(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Are any heads redundant (producing similar outputs)?

    High cosine similarity between head outputs suggests redundancy.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    z_key = f'blocks.{layer}.attn.hook_z'
    z = cache[z_key]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    # Compute mean head output across positions
    head_outputs = []
    for h in range(n_heads):
        out = jnp.einsum('sh,hm->sm', z[:, h, :], W_O[h])  # [seq, d_model]
        mean_out = jnp.mean(out, axis=0)
        head_outputs.append(mean_out)
    head_outputs = jnp.stack(head_outputs)

    norms = jnp.linalg.norm(head_outputs, axis=-1, keepdims=True) + 1e-10
    normed = head_outputs / norms
    sim_matrix = normed @ normed.T

    # Find most redundant pair
    redundancy_threshold = 0.8
    redundant_pairs = []
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            sim = float(sim_matrix[i, j])
            if sim > redundancy_threshold:
                redundant_pairs.append({
                    'head_a': i,
                    'head_b': j,
                    'similarity': sim,
                })

    per_head = []
    for h in range(n_heads):
        max_sim = float(jnp.max(sim_matrix[h, :h])) if h > 0 else 0.0
        if h < n_heads - 1:
            max_sim = max(max_sim, float(jnp.max(sim_matrix[h, h+1:])))
        per_head.append({
            'head': h,
            'output_norm': float(jnp.linalg.norm(head_outputs[h])),
            'max_similarity_to_other': max_sim,
            'is_redundant': max_sim > redundancy_threshold,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'redundant_pairs': redundant_pairs,
        'n_redundant': sum(1 for p in per_head if p['is_redundant']),
    }


def head_output_interference(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Do head outputs destructively interfere (cancel out)?

    Compares the sum of head output norms vs the norm of summed outputs.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    z_key = f'blocks.{layer}.attn.hook_z'
    z = cache[z_key]
    W_O = model.blocks[layer].attn.W_O

    per_position = []
    for pos in range(seq_len):
        individual_norms = []
        total_output = jnp.zeros(model.cfg.d_model)
        for h in range(n_heads):
            out = jnp.einsum('h,hm->m', z[pos, h], W_O[h])
            individual_norms.append(float(jnp.linalg.norm(out)))
            total_output = total_output + out

        sum_of_norms = sum(individual_norms)
        norm_of_sum = float(jnp.linalg.norm(total_output))
        interference_ratio = norm_of_sum / (sum_of_norms + 1e-10)

        per_position.append({
            'position': pos,
            'sum_of_norms': sum_of_norms,
            'norm_of_sum': norm_of_sum,
            'interference_ratio': interference_ratio,
            'has_interference': interference_ratio < 0.8,
        })

    mean_ratio = sum(p['interference_ratio'] for p in per_position) / len(per_position)

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_interference_ratio': mean_ratio,
        'has_significant_interference': mean_ratio < 0.8,
    }


def head_specialization_diversity(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How diverse are the head attention patterns?

    Measures pairwise pattern dissimilarity — high diversity means heads
    attend to different things.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads

    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    patterns = cache[pattern_key]  # [n_heads, seq, seq]

    # Flatten patterns and compute pairwise distances
    flat = patterns.reshape(n_heads, -1)  # [n_heads, seq*seq]
    norms = jnp.linalg.norm(flat, axis=-1, keepdims=True) + 1e-10
    normed = flat / norms
    sim_matrix = normed @ normed.T

    per_head = []
    for h in range(n_heads):
        others = [float(sim_matrix[h, j]) for j in range(n_heads) if j != h]
        mean_sim_to_others = sum(others) / len(others) if others else 0.0

        # Entropy of pattern
        entropy = float(jnp.mean(-jnp.sum(patterns[h] * jnp.log(patterns[h] + 1e-10), axis=-1)))

        per_head.append({
            'head': h,
            'mean_similarity_to_others': mean_sim_to_others,
            'is_unique': mean_sim_to_others < 0.5,
            'pattern_entropy': entropy,
        })

    mean_diversity = 1.0 - float(jnp.mean(sim_matrix[jnp.triu_indices(n_heads, k=1)]))

    return {
        'layer': layer,
        'per_head': per_head,
        'mean_diversity': mean_diversity,
        'n_unique_heads': sum(1 for p in per_head if p['is_unique']),
    }
