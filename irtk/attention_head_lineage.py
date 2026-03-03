"""Attention head lineage: trace how heads compose across layers.

Track which heads in later layers read from which heads in earlier layers,
building a lineage tree of head dependencies.
"""

import jax
import jax.numpy as jnp


def head_to_head_attention(model, tokens, src_layer=0, dst_layer=1):
    """How much does each head in dst_layer attend to outputs of src_layer heads?

    Uses attention patterns to determine information flow between head pairs.

    Returns:
        dict with 'interaction_matrix' [dst_heads, src_heads],
        'strongest_connections' list.
    """
    _, cache = model.run_with_cache(tokens)
    # dst patterns: [n_heads, seq, seq]
    dst_patterns = cache[("pattern", dst_layer)]
    # src z outputs: [seq, n_heads, d_head]
    src_z = cache[("z", src_layer)]
    W_O_src = model.blocks[src_layer].attn.W_O  # [n_heads, d_head, d_model]

    n_dst = dst_patterns.shape[0]
    n_src = src_z.shape[1]

    # Compute src head output norms per position
    # src_output: [n_heads, seq, d_model]
    src_output = jnp.einsum("snh,nhm->nsm", src_z, W_O_src)
    src_norms = jnp.linalg.norm(src_output, axis=-1)  # [n_heads, seq]

    interaction = jnp.zeros((n_dst, n_src))
    for d in range(n_dst):
        pat = dst_patterns[d]  # [seq, seq]
        # weighted sum of src norms by attention pattern
        for s in range(n_src):
            # How much dst head d attends to positions where src head s is active
            weighted = jnp.sum(pat * src_norms[s][None, :])  # scalar
            interaction = interaction.at[d, s].set(weighted)

    # Normalize
    total = jnp.sum(interaction) + 1e-10
    interaction_norm = interaction / total

    strongest = []
    for d in range(n_dst):
        for s in range(n_src):
            strongest.append({
                "dst_head": int(d),
                "src_head": int(s),
                "strength": float(interaction_norm[d, s]),
            })
    strongest.sort(key=lambda x: x["strength"], reverse=True)

    return {
        "interaction_matrix": interaction_norm,
        "strongest_connections": strongest[:10],
    }


def head_output_influence(model, tokens, layer=0):
    """How much does each head's output project onto the next layer's Q/K/V?

    Returns:
        dict with 'per_head' influence on downstream Q, K, V spaces.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    if layer >= n_layers - 1:
        return {"per_head": [], "note": "last layer, no downstream"}

    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    n_heads = z.shape[1]

    # Downstream W_Q, W_K, W_V
    next_layer = layer + 1
    W_Q_next = model.blocks[next_layer].attn.W_Q  # [n_heads, d_model, d_head]
    W_K_next = model.blocks[next_layer].attn.W_K
    W_V_next = model.blocks[next_layer].attn.W_V

    per_head = []
    for h in range(n_heads):
        # head output direction: mean across positions
        head_out = jnp.einsum("sh,hm->sm", z[:, h], W_O[h])  # [seq, d_model]
        mean_out = jnp.mean(head_out, axis=0)  # [d_model]
        mean_out_norm = mean_out / (jnp.linalg.norm(mean_out) + 1e-10)

        # Project onto Q/K/V spaces (mean across next-layer heads)
        q_proj = float(jnp.mean(jnp.abs(mean_out_norm @ W_Q_next.mean(axis=0))))
        k_proj = float(jnp.mean(jnp.abs(mean_out_norm @ W_K_next.mean(axis=0))))
        v_proj = float(jnp.mean(jnp.abs(mean_out_norm @ W_V_next.mean(axis=0))))

        per_head.append({
            "head": int(h),
            "q_influence": q_proj,
            "k_influence": k_proj,
            "v_influence": v_proj,
            "dominant_path": max([("Q", q_proj), ("K", k_proj), ("V", v_proj)], key=lambda x: x[1])[0],
        })
    return {"per_head": per_head}


def head_composition_chain(model, tokens, start_layer=0, start_head=0):
    """Trace the strongest composition chain from a given head forward.

    Returns:
        dict with 'chain' list of (layer, head) steps, 'chain_strength'.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = len(model.blocks)
    chain = [{"layer": start_layer, "head": start_head}]
    current_head = start_head
    strengths = []

    for layer in range(start_layer, n_layers - 1):
        result = head_to_head_attention(model, tokens, src_layer=layer, dst_layer=layer + 1)
        matrix = result["interaction_matrix"]
        # Find which dst head reads most from current head
        next_head = int(jnp.argmax(matrix[:, current_head]))
        strength = float(matrix[next_head, current_head])
        chain.append({"layer": layer + 1, "head": next_head})
        strengths.append(strength)
        current_head = next_head

    total_strength = 1.0
    for s in strengths:
        total_strength *= (s + 1e-10)

    return {
        "chain": chain,
        "per_step_strength": strengths,
        "chain_strength": total_strength,
    }


def layer_head_dependency_graph(model, tokens):
    """Build a full dependency graph across all adjacent layer pairs.

    Returns:
        dict with 'edges' list of (src_layer, src_head, dst_layer, dst_head, strength).
    """
    n_layers = len(model.blocks)
    edges = []
    for layer in range(n_layers - 1):
        result = head_to_head_attention(model, tokens, src_layer=layer, dst_layer=layer + 1)
        for conn in result["strongest_connections"]:
            edges.append({
                "src_layer": layer,
                "src_head": conn["src_head"],
                "dst_layer": layer + 1,
                "dst_head": conn["dst_head"],
                "strength": conn["strength"],
            })
    edges.sort(key=lambda e: e["strength"], reverse=True)
    return {"edges": edges}


def attention_head_lineage_summary(model, tokens):
    """Summary of head lineage: strongest chains and overall connectivity.

    Returns:
        dict with 'strongest_chains', 'n_strong_edges', 'mean_edge_strength'.
    """
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.W_Q.shape[0]

    graph = layer_head_dependency_graph(model, tokens)
    edges = graph["edges"]
    mean_strength = sum(e["strength"] for e in edges) / len(edges) if edges else 0.0
    strong_edges = [e for e in edges if e["strength"] > mean_strength]

    # Trace chains from each head in layer 0
    chains = []
    for h in range(n_heads):
        chain = head_composition_chain(model, tokens, start_layer=0, start_head=h)
        chains.append(chain)
    chains.sort(key=lambda c: c["chain_strength"], reverse=True)

    return {
        "strongest_chains": chains[:3],
        "n_strong_edges": len(strong_edges),
        "mean_edge_strength": mean_strength,
    }
