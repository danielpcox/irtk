"""Cross-position information flow and interaction analysis.

Analyze how information flows between token positions: pairwise influence,
directional transfer, position importance ranking, interaction clustering,
and critical path identification.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Callable


def _get_all_caches(model, tokens):
    """Run model and return full cache."""
    from irtk.hook_points import HookState
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)
    return cache


def pairwise_position_influence(
    model,
    tokens,
    layer: int = 0,
    metric: str = "attention",
) -> dict:
    """Measure pairwise influence between all token positions.

    Uses attention patterns to estimate how much each source position
    influences each destination position.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer to analyze.
        metric: "attention" for attention-based, "residual" for
            residual contribution.

    Returns:
        Dict with influence_matrix [seq, seq], strongest_pairs,
        most_influential_source, most_influenced_target.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)

    if metric == "attention":
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        if pattern_key not in cache:
            return {"error": "Attention pattern not found"}

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
        # Average across heads
        influence = np.mean(patterns, axis=0)  # [seq, seq]
    else:
        # Residual contribution: how much the attention output at each
        # position changes due to each source
        z_key = f"blocks.{layer}.attn.hook_z"
        v_key = f"blocks.{layer}.attn.hook_v"
        pattern_key = f"blocks.{layer}.attn.hook_pattern"

        if pattern_key not in cache or z_key not in cache:
            return {"error": "Required activations not found"}

        patterns = np.array(cache[pattern_key])
        z = np.array(cache[z_key])  # [seq, n_heads, d_head]
        n_heads = patterns.shape[0]

        influence = np.zeros((seq_len, seq_len))
        for h in range(n_heads):
            for dst in range(seq_len):
                total_norm = np.linalg.norm(z[dst, h])
                for src in range(seq_len):
                    influence[dst, src] += patterns[h, dst, src] * total_norm / n_heads

    # Find strongest pairs
    pairs = []
    for dst in range(seq_len):
        for src in range(seq_len):
            if src != dst:
                pairs.append((dst, src, float(influence[dst, src])))
    pairs.sort(key=lambda x: -x[2])

    # Most influential source (highest total outgoing influence)
    source_influence = np.sum(influence, axis=0)  # sum over destinations
    most_influential = int(np.argmax(source_influence))

    # Most influenced target
    target_influence = np.sum(influence, axis=1)  # sum over sources
    most_influenced = int(np.argmax(target_influence))

    return {
        "influence_matrix": jnp.array(influence),
        "strongest_pairs": pairs[:10],
        "most_influential_source": most_influential,
        "most_influenced_target": most_influenced,
        "source_total_influence": jnp.array(source_influence),
        "target_total_influence": jnp.array(target_influence),
    }


def directional_information_transfer(
    model,
    tokens,
    source_pos: int,
    target_pos: int,
    layers: Optional[list] = None,
) -> dict:
    """Trace directional information transfer from source to target.

    Across layers, measures how much information flows from the source
    position to the target via attention.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        source_pos: Source position.
        target_pos: Target position.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with per_layer transfer strength, cumulative_transfer,
        peak_transfer_layer.
    """
    cache = _get_all_caches(model, tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    cumulative = 0.0

    for l in layers:
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
        n_heads = patterns.shape[0]

        # Attention from target to source, averaged across heads
        transfer_weights = patterns[:, target_pos, source_pos]  # [n_heads]
        mean_transfer = float(np.mean(transfer_weights))
        max_transfer = float(np.max(transfer_weights))
        max_head = int(np.argmax(transfer_weights))

        cumulative += mean_transfer
        per_layer.append({
            "layer": l,
            "mean_transfer": mean_transfer,
            "max_transfer": max_transfer,
            "max_transfer_head": max_head,
            "per_head_transfer": transfer_weights.tolist(),
        })

    # Peak transfer layer
    if per_layer:
        peak_layer = max(per_layer, key=lambda x: x["mean_transfer"])["layer"]
    else:
        peak_layer = 0

    return {
        "per_layer": per_layer,
        "cumulative_transfer": cumulative,
        "peak_transfer_layer": peak_layer,
        "source_pos": source_pos,
        "target_pos": target_pos,
    }


def position_importance_ranking(
    model,
    tokens,
    target_pos: int = -1,
    layers: Optional[list] = None,
) -> dict:
    """Rank positions by their importance to the target position.

    Aggregates attention from the target to each source across all
    layers and heads.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        target_pos: Target position to analyze.
        layers: Layers to consider (default: all).

    Returns:
        Dict with position_scores, ranked_positions,
        importance_per_layer matrix.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    importance_per_layer = np.zeros((len(layers), seq_len))

    for li, l in enumerate(layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
        # Average attention from target to each source
        importance_per_layer[li] = np.mean(patterns[:, target_pos, :], axis=0)

    # Aggregate across layers
    position_scores = np.mean(importance_per_layer, axis=0)

    ranked = np.argsort(position_scores)[::-1]
    ranked_positions = [(int(i), float(position_scores[i])) for i in ranked]

    return {
        "position_scores": jnp.array(position_scores),
        "ranked_positions": ranked_positions,
        "importance_per_layer": jnp.array(importance_per_layer),
        "target_pos": target_pos,
    }


def interaction_clustering(
    model,
    tokens,
    layer: int = 0,
    n_clusters: int = 3,
) -> dict:
    """Cluster positions by their interaction patterns.

    Groups positions that attend to similar sets of sources or
    that are attended to by similar sets of targets.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        layer: Layer to analyze.
        n_clusters: Number of clusters.

    Returns:
        Dict with cluster_assignments, cluster_centers,
        cluster_sizes, within_cluster_similarity.
    """
    cache = _get_all_caches(model, tokens)

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache:
        return {"error": "Attention pattern not found"}

    patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
    seq_len = patterns.shape[1]

    # Use average attention pattern as feature for each position
    avg_pattern = np.mean(patterns, axis=0)  # [seq, seq]

    # Simple k-means-like clustering
    n_clusters = min(n_clusters, seq_len)

    # Initialize: spread out initial centers
    indices = np.linspace(0, seq_len - 1, n_clusters, dtype=int)
    centers = avg_pattern[indices].copy()

    for _ in range(20):
        # Assign
        distances = np.zeros((seq_len, n_clusters))
        for c in range(n_clusters):
            distances[:, c] = np.linalg.norm(avg_pattern - centers[c], axis=-1)
        assignments = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for c in range(n_clusters):
            mask = assignments == c
            if np.any(mask):
                new_centers[c] = np.mean(avg_pattern[mask], axis=0)
            else:
                new_centers[c] = centers[c]
        centers = new_centers

    # Compute cluster properties
    cluster_sizes = [int(np.sum(assignments == c)) for c in range(n_clusters)]

    within_sim = []
    for c in range(n_clusters):
        mask = assignments == c
        if np.sum(mask) > 1:
            cluster_points = avg_pattern[mask]
            normed = cluster_points / (np.linalg.norm(cluster_points, axis=-1, keepdims=True) + 1e-10)
            sim = np.mean(normed @ normed.T)
            within_sim.append(float(sim))
        else:
            within_sim.append(1.0)

    return {
        "cluster_assignments": jnp.array(assignments),
        "cluster_centers": jnp.array(centers),
        "cluster_sizes": cluster_sizes,
        "within_cluster_similarity": within_sim,
        "n_clusters": n_clusters,
    }


def critical_information_path(
    model,
    tokens,
    target_pos: int = -1,
    threshold: float = 0.1,
) -> dict:
    """Identify critical information paths to the target position.

    Traces the most important multi-hop paths through the network
    by following high-attention edges across layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        target_pos: Target position.
        threshold: Minimum attention weight to include in path.

    Returns:
        Dict with paths (multi-hop sequences), path_strengths,
        critical_positions, n_active_paths.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    # Build attention graph per layer
    layer_patterns = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache:
            # Average across heads
            avg = np.mean(np.array(cache[pattern_key]), axis=0)  # [seq, seq]
            layer_patterns.append(avg)
        else:
            layer_patterns.append(np.eye(seq_len))

    # Trace paths backward from target_pos
    paths = []

    def trace_back(current_pos, current_layer, path, strength):
        if current_layer < 0:
            paths.append({"path": list(reversed(path)), "strength": strength})
            return

        pattern = layer_patterns[current_layer]
        sources = np.where(pattern[current_pos] > threshold)[0]

        if len(sources) == 0:
            paths.append({"path": list(reversed(path)), "strength": strength})
            return

        for src in sources:
            new_strength = strength * float(pattern[current_pos, src])
            trace_back(src, current_layer - 1, path + [int(src)], new_strength)

    trace_back(target_pos, n_layers - 1, [target_pos], 1.0)

    # Sort by strength and limit
    paths.sort(key=lambda x: -x["strength"])
    paths = paths[:20]

    # Critical positions: appear in the strongest paths
    position_counts = {}
    for p in paths[:10]:
        for pos in p["path"]:
            position_counts[pos] = position_counts.get(pos, 0) + 1

    critical = sorted(position_counts.items(), key=lambda x: -x[1])

    return {
        "paths": paths,
        "path_strengths": [p["strength"] for p in paths],
        "critical_positions": critical[:5],
        "n_active_paths": len(paths),
    }
