"""Attention-based knowledge retrieval analysis.

How attention heads retrieve stored knowledge: query-key matching for
factual recall, value extraction patterns, knowledge routing, retrieval
vs computation classification, and factual association strength.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def _get_all_caches(model, tokens):
    """Run model and return full cache."""
    from irtk.hook_points import HookState
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)
    return cache


def query_key_matching(
    model,
    tokens,
    layer: int = 0,
    head: int = 0,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Analyze query-key matching patterns for knowledge retrieval.

    Examines which source positions the query at `pos` most strongly
    matches with, and characterizes the QK interaction.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer index.
        head: Head index.
        pos: Query position to analyze.
        top_k: Number of top matches.

    Returns:
        Dict with top_matches, match_scores, query_norm, key_norms,
        selectivity (ratio of top match to mean).
    """
    cache = _get_all_caches(model, tokens)

    q_key = f"blocks.{layer}.attn.hook_q"
    k_key = f"blocks.{layer}.attn.hook_k"
    pattern_key = f"blocks.{layer}.attn.hook_pattern"

    q = np.array(cache[q_key][pos, head])  # [d_head]
    k = np.array(cache[k_key][:, head])    # [seq, d_head]
    pattern = np.array(cache[pattern_key][head, pos])  # [seq]

    query_norm = float(np.linalg.norm(q))
    key_norms = np.linalg.norm(k, axis=-1)

    # Raw QK scores (before softmax)
    qk_scores = k @ q / (np.sqrt(q.shape[-1]) + 1e-10)

    # Top matches by attention weight
    top_indices = np.argsort(pattern)[::-1][:top_k]
    top_matches = [(int(i), float(pattern[i]), float(qk_scores[i])) for i in top_indices]

    # Selectivity: how focused is the attention
    selectivity = float(np.max(pattern) / (np.mean(pattern) + 1e-10))

    return {
        "top_matches": top_matches,
        "match_scores": jnp.array(qk_scores),
        "attention_weights": jnp.array(pattern),
        "query_norm": query_norm,
        "key_norms": jnp.array(key_norms),
        "selectivity": selectivity,
    }


def value_extraction_pattern(
    model,
    tokens,
    layer: int = 0,
    head: int = 0,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Analyze how values are extracted and combined by attention.

    For each source position, computes the value vector contribution
    weighted by attention, and identifies what information is extracted.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer index.
        head: Head index.
        pos: Position to analyze.
        top_k: Number of top contributions.

    Returns:
        Dict with per_source value contributions, dominant_source,
        value_diversity (how different source values are).
    """
    cache = _get_all_caches(model, tokens)

    v_key = f"blocks.{layer}.attn.hook_v"
    pattern_key = f"blocks.{layer}.attn.hook_pattern"

    v = np.array(cache[v_key][:, head])     # [seq, d_head]
    pattern = np.array(cache[pattern_key][head, pos])  # [seq]

    per_source = []
    weighted_values = []
    for s in range(len(tokens)):
        weighted_v = pattern[s] * v[s]
        contribution_norm = float(np.linalg.norm(weighted_v))
        weighted_values.append(weighted_v)
        per_source.append({
            "position": s,
            "attention_weight": float(pattern[s]),
            "contribution_norm": contribution_norm,
            "value_norm": float(np.linalg.norm(v[s])),
        })

    per_source.sort(key=lambda x: -x["contribution_norm"])
    dominant_source = per_source[0]["position"] if per_source else 0

    # Value diversity: average cosine distance between source values
    v_normed = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)
    cosines = v_normed @ v_normed.T
    diversity = 1.0 - float(np.mean(np.triu(cosines, k=1)))

    return {
        "per_source": per_source[:top_k],
        "dominant_source": dominant_source,
        "value_diversity": diversity,
        "total_output_norm": float(np.linalg.norm(sum(weighted_values))),
    }


def knowledge_routing(
    model,
    tokens,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Map how knowledge flows through attention heads to the output.

    For each head across all layers, measures the contribution to the
    final residual stream at the given position.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        pos: Position to analyze.
        top_k: Number of top routing heads.

    Returns:
        Dict with per_head contributions, routing_matrix (layers x heads),
        top_routing_heads, total_attention_contribution.
    """
    cache = _get_all_caches(model, tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    routing_matrix = np.zeros((n_layers, n_heads))
    head_contributions = []

    for l in range(n_layers):
        z_key = f"blocks.{l}.attn.hook_z"
        if z_key not in cache:
            continue
        z = np.array(cache[z_key][pos])  # [n_heads, d_head]
        W_O = np.array(model.blocks[l].attn.W_O)  # [n_heads, d_head, d_model]

        for h in range(n_heads):
            output = z[h] @ W_O[h]  # [d_model]
            norm = float(np.linalg.norm(output))
            routing_matrix[l, h] = norm
            head_contributions.append({
                "layer": l,
                "head": h,
                "contribution_norm": norm,
            })

    head_contributions.sort(key=lambda x: -x["contribution_norm"])
    total = float(np.sum(routing_matrix))

    return {
        "routing_matrix": jnp.array(routing_matrix),
        "top_routing_heads": head_contributions[:top_k],
        "total_attention_contribution": total,
        "per_head": head_contributions,
    }


def retrieval_vs_computation(
    model,
    tokens,
    layer: int = 0,
    top_k: int = 3,
) -> dict:
    """Classify heads as retrieval-focused vs computation-focused.

    Retrieval heads: high attention entropy (attend broadly) OR very
    peaked attention to specific positions (look up specific info).
    Computation heads: attention pattern correlates with position
    (positional computation) or shows complex multi-source mixing.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer to analyze.
        top_k: Top entries per category.

    Returns:
        Dict with per_head classification, retrieval_heads,
        computation_heads, classification scores.
    """
    cache = _get_all_caches(model, tokens)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]

    per_head = []
    for h in range(n_heads):
        p = patterns[h]  # [seq, seq]

        # Entropy per query position
        entropies = -np.sum(p * np.log(p + 1e-10), axis=-1)
        mean_entropy = float(np.mean(entropies))
        max_entropy = float(np.log(seq_len))

        # Peakedness: max attention weight averaged over positions
        peakedness = float(np.mean(np.max(p, axis=-1)))

        # Positional correlation: does pattern depend on relative position?
        positions = np.arange(seq_len)
        pos_scores = []
        for q in range(seq_len):
            valid = p[q, :q+1]
            if len(valid) > 1:
                pos_corr = abs(float(np.corrcoef(np.arange(len(valid)), valid)[0, 1]))
                if not np.isnan(pos_corr):
                    pos_scores.append(pos_corr)
        mean_pos_corr = float(np.mean(pos_scores)) if pos_scores else 0.0

        # Classification score: high peakedness + low pos_corr = retrieval
        # high pos_corr = computation (positional)
        retrieval_score = peakedness * (1 - mean_pos_corr)
        computation_score = mean_pos_corr + (1 - peakedness) * 0.5

        per_head.append({
            "head": h,
            "mean_entropy": mean_entropy,
            "peakedness": peakedness,
            "positional_correlation": mean_pos_corr,
            "retrieval_score": retrieval_score,
            "computation_score": computation_score,
            "classification": "retrieval" if retrieval_score > computation_score else "computation",
        })

    retrieval_heads = sorted(
        [h for h in per_head if h["classification"] == "retrieval"],
        key=lambda x: -x["retrieval_score"],
    )[:top_k]
    computation_heads = sorted(
        [h for h in per_head if h["classification"] == "computation"],
        key=lambda x: -x["computation_score"],
    )[:top_k]

    return {
        "per_head": per_head,
        "retrieval_heads": retrieval_heads,
        "computation_heads": computation_heads,
    }


def factual_association_strength(
    model,
    tokens,
    query_pos: int = -1,
    value_pos: Optional[int] = None,
    top_k: int = 5,
) -> dict:
    """Measure the strength of factual associations via attention.

    For each head, measures how strongly the query position attends to
    the value position, and how much that attention changes the output.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        query_pos: Position making the query.
        value_pos: Position holding the value (default: highest attended).
        top_k: Top heads to report.

    Returns:
        Dict with per_head association scores, strongest_associations,
        aggregate_strength.
    """
    cache = _get_all_caches(model, tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    associations = []

    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        z_key = f"blocks.{l}.attn.hook_z"
        v_key = f"blocks.{l}.attn.hook_v"

        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
        z = np.array(cache[z_key])                # [seq, n_heads, d_head]

        for h in range(n_heads):
            p = patterns[h, query_pos]  # [seq]

            # Find value position if not specified
            vp = value_pos if value_pos is not None else int(np.argmax(p))

            attention_weight = float(p[vp])
            output_norm = float(np.linalg.norm(z[query_pos, h]))

            # Association strength = attention weight * output magnitude
            strength = attention_weight * output_norm

            associations.append({
                "layer": l,
                "head": h,
                "value_position": vp,
                "attention_weight": attention_weight,
                "output_norm": output_norm,
                "association_strength": strength,
            })

    associations.sort(key=lambda x: -x["association_strength"])
    aggregate = float(sum(a["association_strength"] for a in associations))

    return {
        "per_head": associations,
        "strongest_associations": associations[:top_k],
        "aggregate_strength": aggregate,
    }
