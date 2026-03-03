"""Attention motif discovery.

Discovers recurring attention patterns (motifs) across heads and layers.
Goes beyond head classification to find abstract structural patterns and
their functional roles.

Functions:
- extract_attention_motifs: Cluster attention patterns into motif types
- motif_prevalence_analysis: Which layers/heads use each motif
- motif_input_dependency: What token properties trigger each motif
- motif_function_inference: Downstream effect of each motif
- motif_diversity_by_layer: How diverse attention patterns are per layer

References:
    - Olsson et al. (2022) "In-context Learning and Induction Heads"
    - Voita et al. (2019) "Analyzing Multi-Head Self-Attention"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def extract_attention_motifs(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    n_motifs: int = 4,
) -> dict:
    """Cluster attention patterns into motif types.

    Uses SVD-based factorization of the stacked attention patterns
    to discover recurring motifs.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        n_motifs: Number of motif clusters.

    Returns:
        Dict with:
            "motif_patterns": [n_motifs, seq, seq] representative patterns
            "head_motif_assignments": [n_layers, n_heads] motif index per head
            "head_motif_scores": [n_layers, n_heads, n_motifs] similarity to each motif
            "n_motifs_found": actual number of distinct motifs
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    # Collect all attention patterns
    patterns = []
    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key in cache.cache_dict:
            pat = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
            for h in range(n_heads):
                patterns.append(pat[h].flatten())

    if not patterns:
        return {
            "motif_patterns": np.zeros((n_motifs, seq_len, seq_len)),
            "head_motif_assignments": np.zeros((n_layers, n_heads), dtype=int),
            "head_motif_scores": np.zeros((n_layers, n_heads, n_motifs)),
            "n_motifs_found": 0,
        }

    patterns = np.array(patterns)  # [n_total_heads, seq*seq]
    n_total = len(patterns)
    n_motifs = min(n_motifs, n_total)

    # SVD to find principal attention patterns
    U, S, Vt = np.linalg.svd(patterns, full_matrices=False)
    motif_vecs = Vt[:n_motifs]  # [n_motifs, seq*seq]

    # Reshape to attention pattern format
    motif_patterns = motif_vecs.reshape(n_motifs, seq_len, seq_len)

    # Assign each head to nearest motif
    assignments = np.zeros((n_layers, n_heads), dtype=int)
    scores = np.zeros((n_layers, n_heads, n_motifs))

    idx = 0
    for l in range(n_layers):
        for h in range(n_heads):
            if idx < n_total:
                for m in range(n_motifs):
                    norm_p = np.linalg.norm(patterns[idx])
                    norm_m = np.linalg.norm(motif_vecs[m])
                    if norm_p > 1e-10 and norm_m > 1e-10:
                        scores[l, h, m] = abs(float(
                            np.dot(patterns[idx], motif_vecs[m]) / (norm_p * norm_m)
                        ))
                assignments[l, h] = int(np.argmax(scores[l, h]))
            idx += 1

    return {
        "motif_patterns": motif_patterns,
        "head_motif_assignments": assignments,
        "head_motif_scores": scores,
        "n_motifs_found": n_motifs,
    }


def motif_prevalence_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    n_motifs: int = 4,
) -> dict:
    """Which layers/heads use each motif and how prevalent each is.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        n_motifs: Number of motifs to discover.

    Returns:
        Dict with:
            "motif_counts": [n_motifs] number of heads using each motif
            "motif_by_layer": [n_layers, n_motifs] count of motif usage per layer
            "dominant_motif": most common motif index
            "motif_diversity": entropy of motif distribution
    """
    result = extract_attention_motifs(model, tokens, n_motifs=n_motifs)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    assignments = result["head_motif_assignments"]

    counts = np.zeros(n_motifs, dtype=int)
    by_layer = np.zeros((n_layers, n_motifs), dtype=int)

    for l in range(n_layers):
        for h in range(n_heads):
            m = assignments[l, h]
            counts[m] += 1
            by_layer[l, m] += 1

    dominant = int(np.argmax(counts))

    # Entropy
    total = float(np.sum(counts))
    if total > 0:
        probs = counts / total
        probs = probs[probs > 0]
        diversity = -float(np.sum(probs * np.log(probs + 1e-10)))
    else:
        diversity = 0.0

    return {
        "motif_counts": counts,
        "motif_by_layer": by_layer,
        "dominant_motif": dominant,
        "motif_diversity": diversity,
    }


def motif_input_dependency(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
) -> dict:
    """What token properties trigger a head's attention pattern.

    Measures which source positions get the most attention and whether
    the pattern depends on token identity or position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer to analyze.
        head: Head to analyze.

    Returns:
        Dict with:
            "position_concentration": [seq_len] mean attention received per position
            "diagonal_strength": how much attention falls on the diagonal (self-attention)
            "recency_strength": how much attention falls on recent positions
            "uniformity": how uniform the attention pattern is (1 = uniform)
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache.cache_dict:
        return {
            "position_concentration": np.zeros(seq_len),
            "diagonal_strength": 0.0,
            "recency_strength": 0.0,
            "uniformity": 0.0,
        }

    pattern = np.array(cache.cache_dict[key][head])  # [seq, seq]

    # Mean attention received per position (averaged over queries)
    pos_conc = np.mean(pattern, axis=0)

    # Diagonal strength
    diag_vals = np.diag(pattern)
    diag_strength = float(np.mean(diag_vals))

    # Recency: for each query, how much attention goes to the last 2 positions before it
    recency_total = 0.0
    count = 0
    for q in range(2, seq_len):
        recency_total += float(pattern[q, q-1] + pattern[q, q-2])
        count += 1
    recency = recency_total / (2 * max(count, 1))

    # Uniformity: 1 - (std / mean) for attention weights
    flat = pattern[pattern > 0]
    if len(flat) > 0 and np.mean(flat) > 1e-10:
        uniformity = 1.0 - float(np.std(flat) / np.mean(flat))
        uniformity = max(0.0, uniformity)
    else:
        uniformity = 0.0

    return {
        "position_concentration": pos_conc,
        "diagonal_strength": diag_strength,
        "recency_strength": recency,
        "uniformity": uniformity,
    }


def motif_function_inference(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
) -> dict:
    """Downstream effect of a head's attention pattern.

    Measures what the head contributes to the output logits and how
    its pattern relates to its function.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer to analyze.
        head: Head to analyze.

    Returns:
        Dict with:
            "logit_contribution_norm": L2 norm of head's logit contribution
            "top_promoted_tokens": list of (token_idx, logit_value) for top-5
            "top_demoted_tokens": list of (token_idx, logit_value) for bottom-5
            "output_direction_norm": norm of the head's output vector
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache.cache_dict:
        return {
            "logit_contribution_norm": 0.0,
            "top_promoted_tokens": [],
            "top_demoted_tokens": [],
            "output_direction_norm": 0.0,
        }

    z = np.array(cache.cache_dict[z_key])  # [seq, n_heads, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O[head])  # [d_head, d_model]

    head_out = z[pos, head] @ W_O  # [d_model]
    head_logits = head_out @ W_U  # [d_vocab]

    norm = float(np.linalg.norm(head_logits))
    out_norm = float(np.linalg.norm(head_out))

    top_idx = np.argsort(head_logits)[::-1][:5]
    bot_idx = np.argsort(head_logits)[:5]

    top_promoted = [(int(i), float(head_logits[i])) for i in top_idx]
    top_demoted = [(int(i), float(head_logits[i])) for i in bot_idx]

    return {
        "logit_contribution_norm": norm,
        "top_promoted_tokens": top_promoted,
        "top_demoted_tokens": top_demoted,
        "output_direction_norm": out_norm,
    }


def motif_diversity_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """How diverse attention patterns are within each layer.

    Measures pairwise cosine distance between heads within a layer
    to see if heads specialize or duplicate.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "layer_diversity": [n_layers] mean pairwise distance within each layer
            "most_diverse_layer": layer with most diverse head patterns
            "least_diverse_layer": layer with most similar head patterns
            "head_similarities": [n_layers] mean pairwise cosine similarity
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    diversity = np.zeros(n_layers)
    similarities = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key not in cache.cache_dict:
            continue

        pattern = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
        flat = pattern.reshape(n_heads, -1)
        norms = np.linalg.norm(flat, axis=1)

        pair_sims = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                if norms[i] > 1e-10 and norms[j] > 1e-10:
                    cos = float(np.dot(flat[i], flat[j]) / (norms[i] * norms[j]))
                    pair_sims.append(cos)

        if pair_sims:
            mean_sim = float(np.mean(pair_sims))
            similarities[l] = mean_sim
            diversity[l] = 1.0 - mean_sim

    most_div = int(np.argmax(diversity))
    least_div = int(np.argmin(diversity))

    return {
        "layer_diversity": diversity,
        "most_diverse_layer": most_div,
        "least_diverse_layer": least_div,
        "head_similarities": similarities,
    }
