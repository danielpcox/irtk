"""Attention pattern interpretation: detect and classify attention motifs."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def detect_attention_motifs(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Classify what kind of attention pattern a head is using.

    Detects: diagonal (previous-token), identity (self-attention),
    BOS (first-token), uniform, position-based, and content-based patterns.
    """
    _, cache = model.run_with_cache(tokens)
    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern = cache[pattern_key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Diagonal detection (previous-token): high weight on position i-1
    if seq_len > 1:
        diag_weights = jnp.array([float(pattern[i, i-1]) for i in range(1, seq_len)])
        diagonal_score = float(jnp.mean(diag_weights))
    else:
        diagonal_score = 0.0

    # Identity detection (self-attention): high weight on position i
    identity_weights = jnp.diag(pattern)
    identity_score = float(jnp.mean(identity_weights))

    # BOS detection: high weight on position 0
    bos_weights = pattern[:, 0]
    bos_score = float(jnp.mean(bos_weights[1:])) if seq_len > 1 else 0.0

    # Uniform detection: low entropy difference from uniform
    uniform_pattern = jnp.tril(jnp.ones_like(pattern))
    uniform_pattern = uniform_pattern / jnp.sum(uniform_pattern, axis=-1, keepdims=True)
    uniform_diff = float(jnp.mean(jnp.abs(pattern - uniform_pattern)))
    uniform_score = max(0.0, 1.0 - uniform_diff * 5)

    # Determine dominant motif
    scores = {
        'previous_token': diagonal_score,
        'self_attention': identity_score,
        'bos_attention': bos_score,
        'uniform': uniform_score,
    }
    dominant = max(scores, key=scores.get)

    return {
        'layer': layer,
        'head': head,
        'dominant_motif': dominant,
        'motif_scores': scores,
        'diagonal_score': diagonal_score,
        'identity_score': identity_score,
        'bos_score': bos_score,
        'uniform_score': uniform_score,
    }


def attention_pattern_summary(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Summarize an attention pattern with statistics.

    Computes entropy, sparsity, mean/max attention, attended positions.
    """
    _, cache = model.run_with_cache(tokens)
    pattern_key = f'blocks.{layer}.attn.hook_pattern'
    pattern = cache[pattern_key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Per-position entropy
    entropies = -jnp.sum(pattern * jnp.log(pattern + 1e-10), axis=-1)
    mean_entropy = float(jnp.mean(entropies))
    max_entropy = float(jnp.log(jnp.arange(1, seq_len + 1, dtype=jnp.float32)).mean())

    # Sparsity: fraction of attention above threshold
    threshold = 1.0 / seq_len
    sparse_frac = float(jnp.mean(pattern > threshold * 2))

    # Mean attention distance
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    distances = []
    for i in range(seq_len):
        attended = positions[:i+1]
        weights = pattern[i, :i+1]
        mean_pos = jnp.sum(weights * attended)
        dist = float(i - mean_pos)
        distances.append(dist)
    mean_distance = sum(distances) / len(distances) if distances else 0.0

    # Max attention per position
    max_attn = float(jnp.mean(jnp.max(pattern, axis=-1)))

    return {
        'layer': layer,
        'head': head,
        'mean_entropy': mean_entropy,
        'sparsity': sparse_frac,
        'mean_distance': mean_distance,
        'mean_max_attention': max_attn,
        'is_sparse': mean_entropy < max_entropy * 0.5,
    }


def head_function_profile(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """Profile what a head does to the residual stream.

    Measures output norm, logit contribution, and direction consistency.
    """
    logits, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    # Get head's output contribution
    z_key = f'blocks.{layer}.attn.hook_z'
    z = cache[z_key]  # [seq, n_heads, d_head]
    head_z = z[:, head, :]  # [seq, d_head]

    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    head_output = jnp.einsum('sh,hm->sm', head_z, W_O[head])  # [seq, d_model]

    # Output norms
    output_norms = jnp.linalg.norm(head_output, axis=-1)
    mean_norm = float(jnp.mean(output_norms))

    # Logit contribution via unembed
    W_U = model.unembed.W_U
    logit_contrib = head_output @ W_U  # [seq, d_vocab]
    mean_logit_mag = float(jnp.mean(jnp.abs(logit_contrib)))

    # Direction consistency (pairwise cosine)
    normed = head_output / (jnp.linalg.norm(head_output, axis=-1, keepdims=True) + 1e-10)
    mean_dir = jnp.mean(normed, axis=0)
    mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)
    alignments = normed @ mean_dir
    direction_consistency = float(jnp.mean(alignments))

    return {
        'layer': layer,
        'head': head,
        'mean_output_norm': mean_norm,
        'mean_logit_magnitude': mean_logit_mag,
        'direction_consistency': direction_consistency,
        'is_consistent': direction_consistency > 0.5,
    }


def all_heads_motif_classification(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Classify all heads in the model by their dominant motif.

    Returns a summary of how many heads of each type exist.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head = []
    motif_counts = {}

    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        pattern_all = cache[pattern_key]  # [n_heads, seq, seq]
        seq_len = pattern_all.shape[1]

        for head in range(n_heads):
            pattern = pattern_all[head]

            # Quick motif scores
            if seq_len > 1:
                diag = float(jnp.mean(jnp.array([pattern[i, i-1] for i in range(1, seq_len)])))
            else:
                diag = 0.0
            identity = float(jnp.mean(jnp.diag(pattern)))
            bos = float(jnp.mean(pattern[1:, 0])) if seq_len > 1 else 0.0

            scores = {'previous_token': diag, 'self_attention': identity, 'bos_attention': bos}
            dominant = max(scores, key=scores.get)

            per_head.append({
                'layer': layer,
                'head': head,
                'dominant_motif': dominant,
                'confidence': scores[dominant],
            })
            motif_counts[dominant] = motif_counts.get(dominant, 0) + 1

    return {
        'per_head': per_head,
        'motif_counts': motif_counts,
        'n_heads_total': n_layers * n_heads,
    }


def attention_pattern_evolution(model: HookedTransformer, tokens: jnp.ndarray, head: int) -> dict:
    """Track how a specific head index's pattern changes across layers.

    Useful for understanding if head N plays different roles at different depths.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    prev_pattern = None

    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        pattern = cache[pattern_key][head]
        seq_len = pattern.shape[0]

        entropy = float(jnp.mean(-jnp.sum(pattern * jnp.log(pattern + 1e-10), axis=-1)))
        max_attn = float(jnp.mean(jnp.max(pattern, axis=-1)))

        if prev_pattern is not None:
            similarity = float(jnp.mean(jnp.sum(prev_pattern * pattern, axis=-1)))
        else:
            similarity = 1.0

        per_layer.append({
            'layer': layer,
            'mean_entropy': entropy,
            'mean_max_attention': max_attn,
            'similarity_to_previous': similarity,
        })
        prev_pattern = pattern

    # Overall stability
    similarities = [p['similarity_to_previous'] for p in per_layer[1:]]
    mean_stability = sum(similarities) / len(similarities) if similarities else 1.0

    return {
        'head': head,
        'per_layer': per_layer,
        'mean_stability': mean_stability,
        'is_stable': mean_stability > 0.5,
    }
