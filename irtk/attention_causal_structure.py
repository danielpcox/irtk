"""Attention causal structure: analyze causal dependencies in attention patterns.

Detect causal chains, information bottlenecks, multi-hop paths,
and causal influence strength across positions.
"""

import jax
import jax.numpy as jnp


def causal_attention_chain(model, tokens, target_position=-1):
    """Trace the causal chain of attention back from a target position.

    For each layer, find which source positions have the most attention,
    building a chain of influence.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if target_position < 0:
        target_position = len(tokens) + target_position

    per_layer = []
    current_positions = {target_position: 1.0}

    for layer in range(n_layers - 1, -1, -1):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        # Average across heads
        avg_attn = jnp.mean(pattern, axis=0)  # [seq, seq]

        # For each currently tracked position, find its sources
        new_positions = {}
        for pos, weight in current_positions.items():
            attn_row = avg_attn[pos]  # [seq]
            # Top sources
            top_k = min(3, len(tokens))
            top_indices = jnp.argsort(attn_row)[-top_k:][::-1]
            for idx in top_indices:
                src = int(idx)
                contrib = weight * float(attn_row[idx])
                if src in new_positions:
                    new_positions[src] = max(new_positions[src], contrib)
                else:
                    new_positions[src] = contrib

        per_layer.append({
            "layer": layer,
            "tracked_positions": dict(current_positions),
            "n_active_positions": len(current_positions),
        })
        current_positions = new_positions

    # Reverse to layer order
    per_layer = per_layer[::-1]

    return {
        "per_layer": per_layer,
        "target_position": target_position,
        "root_positions": sorted(current_positions.keys()),
        "n_root_positions": len(current_positions),
    }


def attention_information_bottleneck(model, tokens):
    """Detect positions that act as information bottlenecks.

    A bottleneck position is one that receives attention from many positions
    and also sends attention to many positions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    per_position = []
    for pos in range(seq_len):
        total_incoming = 0.0
        total_outgoing = 0.0
        for layer in range(n_layers):
            key = f"blocks.{layer}.attn.hook_pattern"
            if key not in cache:
                continue
            pattern = cache[key]
            avg_attn = jnp.mean(pattern, axis=0)
            # Incoming: how much other positions attend to this one
            total_incoming += float(jnp.sum(avg_attn[:, pos]))
            # Outgoing: how much this position attends to others
            total_outgoing += float(jnp.sum(avg_attn[pos, :]))

        per_position.append({
            "position": pos,
            "token_id": int(tokens[pos]),
            "total_incoming": total_incoming,
            "total_outgoing": total_outgoing,
            "bottleneck_score": total_incoming * total_outgoing,
        })

    # Sort by bottleneck score
    per_position.sort(key=lambda p: p["bottleneck_score"], reverse=True)
    max_score = per_position[0]["bottleneck_score"] if per_position else 1.0

    return {
        "per_position": per_position,
        "top_bottleneck_position": per_position[0]["position"] if per_position else 0,
        "has_clear_bottleneck": per_position[0]["bottleneck_score"] > 2 * per_position[-1]["bottleneck_score"] if len(per_position) > 1 else False,
    }


def multi_hop_attention_paths(model, tokens, source, target):
    """Find multi-hop attention paths from source to target position.

    Computes attention flow across layers via matrix multiplication.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Build per-layer average attention matrices
    attn_matrices = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]
        avg_attn = jnp.mean(pattern, axis=0)  # [seq, seq]
        attn_matrices.append(avg_attn)

    # Compute cumulative attention flow
    per_depth = []
    flow = attn_matrices[0] if attn_matrices else jnp.eye(len(tokens))

    for depth, mat in enumerate(attn_matrices):
        if depth == 0:
            flow = mat
        else:
            flow = mat @ flow
        path_strength = float(flow[target, source])
        per_depth.append({
            "depth": depth + 1,
            "path_strength": path_strength,
        })

    return {
        "per_depth": per_depth,
        "source": source,
        "target": target,
        "max_path_strength": max(p["path_strength"] for p in per_depth) if per_depth else 0.0,
        "best_depth": max(per_depth, key=lambda p: p["path_strength"])["depth"] if per_depth else 0,
    }


def causal_influence_matrix(model, tokens, layer=0):
    """Compute the causal influence between all position pairs at a layer.

    Combines attention patterns with value norms for a richer picture.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache:
        return {"influence_matrix": jnp.zeros((seq_len, seq_len)), "layer": layer, "mean_influence": 0.0}

    pattern = cache[key]  # [n_heads, seq, seq]

    # Get value norms
    v_key = f"blocks.{layer}.attn.hook_v"
    if v_key in cache:
        v = cache[v_key]  # [seq, n_heads, d_head]
        v_norms = jnp.linalg.norm(v, axis=-1)  # [seq, n_heads]
        # Weight attention by value norms
        influence = jnp.zeros((seq_len, seq_len))
        for h in range(n_heads):
            weighted = pattern[h] * v_norms[:, h][None, :]  # [seq, seq]
            influence = influence + weighted
        influence = influence / n_heads
    else:
        influence = jnp.mean(pattern, axis=0)

    mean_influence = float(jnp.mean(influence))

    return {
        "influence_matrix": influence,
        "layer": layer,
        "mean_influence": mean_influence,
        "max_influence": float(jnp.max(influence)),
    }


def causal_structure_summary(model, tokens):
    """Cross-layer summary of causal attention structure.

    Tracks attention concentration, dominant sources, and flow patterns.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]
        avg_attn = jnp.mean(pattern, axis=0)  # [seq, seq]

        # Concentration: entropy of attention
        entropy = -jnp.sum(avg_attn * jnp.log(avg_attn + 1e-10), axis=-1)
        mean_entropy = float(jnp.mean(entropy))

        # Dominant source per position
        dominant_sources = jnp.argmax(avg_attn, axis=-1)
        # How often is position 0 (BOS) dominant?
        bos_dominant_frac = float(jnp.mean(dominant_sources == 0))

        # Self-attention fraction
        self_attn = float(jnp.mean(jnp.diag(avg_attn)))

        per_layer.append({
            "layer": layer,
            "mean_entropy": mean_entropy,
            "bos_dominant_fraction": bos_dominant_frac,
            "self_attention_fraction": self_attn,
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
        "overall_pattern": "concentrated" if per_layer and per_layer[-1]["mean_entropy"] < per_layer[0]["mean_entropy"] else "distributed",
    }
