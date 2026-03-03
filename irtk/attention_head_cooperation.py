"""Attention head cooperation: how heads work together within and across layers.

Detect complementary heads, redundant heads, head pipelines,
and cooperative patterns in attention.
"""

import jax
import jax.numpy as jnp


def within_layer_cooperation(model, tokens, layer=0, position=-1):
    """Analyze how heads within a layer cooperate or compete.

    Measures output alignment and attention pattern overlap.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    # Get per-head outputs
    z_key = f"blocks.{layer}.attn.hook_z"
    pattern_key = f"blocks.{layer}.attn.hook_pattern"

    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    pairs = []
    head_outputs = []

    if z_key in cache and pattern_key in cache:
        z = cache[z_key]  # [seq, n_heads, d_head]
        pattern = cache[pattern_key]  # [n_heads, seq, seq]

        for h in range(n_heads):
            out = z[position, h] @ W_O[h]  # [d_model]
            head_outputs.append(out)

        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                # Output alignment
                ni = jnp.linalg.norm(head_outputs[i]) + 1e-10
                nj = jnp.linalg.norm(head_outputs[j]) + 1e-10
                output_cos = float(jnp.dot(head_outputs[i], head_outputs[j]) / (ni * nj))

                # Attention pattern overlap
                pi = pattern[i, position]
                pj = pattern[j, position]
                attn_cos = float(jnp.dot(pi, pj) / (jnp.linalg.norm(pi) * jnp.linalg.norm(pj) + 1e-10))

                # Classification
                if output_cos > 0.5 and attn_cos > 0.5:
                    relation = "redundant"
                elif output_cos < -0.3:
                    relation = "competing"
                elif attn_cos < 0.3:
                    relation = "complementary"
                else:
                    relation = "independent"

                pairs.append({
                    "head_i": i, "head_j": j,
                    "output_cosine": output_cos,
                    "attention_cosine": attn_cos,
                    "relation": relation,
                })

    return {
        "pairs": pairs,
        "layer": layer,
        "n_redundant": sum(1 for p in pairs if p["relation"] == "redundant"),
        "n_complementary": sum(1 for p in pairs if p["relation"] == "complementary"),
    }


def cross_layer_head_pipeline(model, tokens, position=-1):
    """Detect head pipelines where one layer's heads feed into another's.

    Measures how much layer L's output directions align with layer L+1's Q/K.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    per_transition = []
    for layer in range(n_layers - 1):
        result_key = f"blocks.{layer}.attn.hook_result"
        next_pattern_key = f"blocks.{layer + 1}.attn.hook_pattern"

        if result_key in cache and next_pattern_key in cache:
            # Layer L output
            result = cache[result_key][position]  # [d_model]
            result_norm = jnp.linalg.norm(result) + 1e-10

            # Layer L+1 attention pattern change
            next_pattern = cache[next_pattern_key]  # [n_heads, seq, seq]
            avg_next = jnp.mean(next_pattern[:, position, :], axis=0)
            # How concentrated is next layer's attention?
            entropy = -float(jnp.sum(avg_next * jnp.log(avg_next + 1e-10)))

            per_transition.append({
                "from_layer": layer,
                "to_layer": layer + 1,
                "output_norm": float(result_norm),
                "next_layer_entropy": entropy,
            })

    return {
        "per_transition": per_transition,
        "position": position,
        "has_pipeline": len(per_transition) > 0,
    }


def head_output_diversity(model, tokens, layer=0):
    """Measure how diverse head outputs are across all positions.

    Diverse outputs = heads contribute different information.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    z_key = f"blocks.{layer}.attn.hook_z"
    W_O = model.blocks[layer].attn.W_O

    if z_key not in cache:
        return {"per_head": [], "mean_diversity": 0.0, "layer": layer}

    z = cache[z_key]  # [seq, n_heads, d_head]

    # Compute output for each head across all positions
    per_head = []
    head_outputs = []
    for h in range(n_heads):
        out = z[:, h, :] @ W_O[h]  # [seq, d_model]
        norm = float(jnp.linalg.norm(out))
        head_outputs.append(out.reshape(-1))
        per_head.append({
            "head": h,
            "output_norm": norm,
        })

    # Pairwise diversity
    cosines = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            ni = jnp.linalg.norm(head_outputs[i]) + 1e-10
            nj = jnp.linalg.norm(head_outputs[j]) + 1e-10
            cos = float(jnp.dot(head_outputs[i], head_outputs[j]) / (ni * nj))
            cosines.append(cos)

    mean_cos = sum(cosines) / max(len(cosines), 1)

    return {
        "per_head": per_head,
        "mean_pairwise_cosine": mean_cos,
        "mean_diversity": 1.0 - abs(mean_cos),
        "layer": layer,
        "is_diverse": abs(mean_cos) < 0.5,
    }


def head_contribution_ranking(model, tokens, position=-1):
    """Rank heads by their contribution to the final prediction.

    Projects each head's output through the unembedding matrix.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    all_heads = []
    for layer in range(n_layers):
        z_key = f"blocks.{layer}.attn.hook_z"
        W_O = model.blocks[layer].attn.W_O

        if z_key not in cache:
            continue
        z = cache[z_key]

        for h in range(n_heads):
            out = z[position, h] @ W_O[h]  # [d_model]
            logits = out @ W_U + b_U  # [vocab]
            logit_range = float(jnp.max(logits) - jnp.min(logits))
            top_token = int(jnp.argmax(logits))

            all_heads.append({
                "layer": layer,
                "head": h,
                "logit_range": logit_range,
                "top_promoted_token": top_token,
                "output_norm": float(jnp.linalg.norm(out)),
            })

    all_heads.sort(key=lambda x: x["logit_range"], reverse=True)

    return {
        "heads": all_heads,
        "position": position,
        "most_impactful": f"L{all_heads[0]['layer']}H{all_heads[0]['head']}" if all_heads else "none",
    }


def cooperation_summary(model, tokens):
    """Cross-layer summary of head cooperation patterns.

    Combines diversity, redundancy, and contribution metrics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        div = head_output_diversity(model, tokens, layer=layer)
        coop = within_layer_cooperation(model, tokens, layer=layer)

        per_layer.append({
            "layer": layer,
            "mean_diversity": div["mean_diversity"],
            "n_redundant_pairs": coop["n_redundant"],
            "n_complementary_pairs": coop["n_complementary"],
            "is_diverse": div["is_diverse"],
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
    }
