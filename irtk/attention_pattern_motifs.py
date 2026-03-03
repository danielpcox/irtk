"""Attention pattern motifs: detect common attention pattern structures.

Identify diagonal (previous-token), stripe (uniform), block, triangular,
and other structural motifs in attention patterns.
"""

import jax
import jax.numpy as jnp


def detect_diagonal_motif(model, tokens, layer=0, head=0):
    """Detect diagonal/previous-token attention pattern.

    Measures how much attention follows the subdiagonal (attending to i-1).
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache:
        return {"diagonal_score": 0.0, "is_diagonal": False, "layer": layer, "head": head}

    pattern = cache[key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Subdiagonal mass
    diag_mass = 0.0
    for i in range(1, seq_len):
        diag_mass += float(pattern[i, i - 1])
    avg_diag = diag_mass / max(seq_len - 1, 1)

    # Self-diagonal mass
    self_mass = float(jnp.mean(jnp.diag(pattern)))

    return {
        "diagonal_score": avg_diag,
        "self_attention_score": self_mass,
        "is_diagonal": avg_diag > 0.3,
        "is_self_attention": self_mass > 0.3,
        "layer": layer,
        "head": head,
    }


def detect_stripe_motif(model, tokens, layer=0, head=0):
    """Detect stripe/uniform attention pattern.

    A stripe pattern means a specific key position receives high attention
    from many query positions (column-dominant).
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache:
        return {"stripe_score": 0.0, "dominant_column": 0, "layer": layer, "head": head}

    pattern = cache[key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Column sums (how much each key position is attended to)
    col_sums = jnp.sum(pattern, axis=0)  # [seq]
    max_col = int(jnp.argmax(col_sums))
    max_col_mass = float(col_sums[max_col]) / seq_len

    # Entropy of column distribution
    col_dist = col_sums / (jnp.sum(col_sums) + 1e-10)
    entropy = -float(jnp.sum(col_dist * jnp.log(col_dist + 1e-10)))
    max_entropy = float(jnp.log(seq_len))
    concentration = 1.0 - entropy / (max_entropy + 1e-10)

    return {
        "stripe_score": max_col_mass,
        "column_concentration": concentration,
        "dominant_column": max_col,
        "is_stripe": max_col_mass > 0.3,
        "layer": layer,
        "head": head,
    }


def detect_block_motif(model, tokens, layer=0, head=0):
    """Detect block/local attention pattern.

    A block pattern means tokens attend mostly to nearby positions.
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache:
        return {"local_score": 0.0, "layer": layer, "head": head}

    pattern = cache[key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Local attention: attention within distance k
    k = max(2, seq_len // 4)
    local_mass = 0.0
    total_mass = 0.0
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # Causal
                total_mass += float(pattern[i, j])
                if abs(i - j) <= k:
                    local_mass += float(pattern[i, j])

    local_score = local_mass / (total_mass + 1e-10)

    return {
        "local_score": local_score,
        "window_size": k,
        "is_local": local_score > 0.7,
        "layer": layer,
        "head": head,
    }


def detect_triangular_motif(model, tokens, layer=0, head=0):
    """Detect triangular/causal attention with specific structure.

    Measures how much the pattern follows a triangular (uniform causal) shape.
    """
    _, cache = model.run_with_cache(tokens)
    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache:
        return {"triangular_score": 0.0, "layer": layer, "head": head}

    pattern = cache[key][head]  # [seq, seq]
    seq_len = pattern.shape[0]

    # Ideal uniform causal: each position attends uniformly to all previous
    # Cosine between actual pattern and uniform causal
    ideal = jnp.zeros_like(pattern)
    for i in range(seq_len):
        if i > 0:
            ideal = ideal.at[i, :i + 1].set(1.0 / (i + 1))
        else:
            ideal = ideal.at[0, 0].set(1.0)

    actual_flat = pattern.reshape(-1)
    ideal_flat = ideal.reshape(-1)
    cos = float(jnp.dot(actual_flat, ideal_flat) /
                (jnp.linalg.norm(actual_flat) * jnp.linalg.norm(ideal_flat) + 1e-10))

    return {
        "triangular_score": cos,
        "is_triangular": cos > 0.7,
        "layer": layer,
        "head": head,
    }


def attention_motif_summary(model, tokens):
    """Classify all heads by their dominant attention motif.

    Categories: diagonal, stripe, local, triangular, mixed.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    classifications = []
    for layer in range(n_layers):
        for head in range(n_heads):
            diag = detect_diagonal_motif(model, tokens, layer, head)
            stripe = detect_stripe_motif(model, tokens, layer, head)
            block = detect_block_motif(model, tokens, layer, head)
            tri = detect_triangular_motif(model, tokens, layer, head)

            scores = {
                "diagonal": diag["diagonal_score"],
                "stripe": stripe["stripe_score"],
                "local": block["local_score"],
                "triangular": tri["triangular_score"],
            }
            dominant = max(scores, key=scores.get)

            classifications.append({
                "layer": layer,
                "head": head,
                "dominant_motif": dominant,
                "scores": scores,
            })

    # Count motifs
    motif_counts = {}
    for c in classifications:
        m = c["dominant_motif"]
        motif_counts[m] = motif_counts.get(m, 0) + 1

    return {
        "classifications": classifications,
        "motif_counts": motif_counts,
        "most_common_motif": max(motif_counts, key=motif_counts.get) if motif_counts else "none",
    }
