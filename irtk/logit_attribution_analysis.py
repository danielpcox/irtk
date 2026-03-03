"""Logit attribution analysis: fine-grained logit attribution to every component.

Decompose the final logit into contributions from each layer's attention
and MLP, embed/unembed bias, and per-head contributions.
"""

import jax
import jax.numpy as jnp


def full_logit_decomposition(model, tokens, position=-1, target_token=None):
    """Decompose a target token's logit into contributions from every component.

    Attributes logit to: embed, each layer's attention, each layer's MLP, and unembed bias.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    if position < 0:
        position = seq_len + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    if target_token is None:
        # Use the model's top prediction
        final_key = f"blocks.{n_layers - 1}.hook_resid_post"
        if final_key not in cache:
            final_key = f"blocks.{n_layers - 1}.hook_resid_pre"
        logits = cache[final_key][position] @ W_U + b_U
        target_token = int(jnp.argmax(logits))

    unembed_dir = W_U[:, target_token]  # [d_model]
    bias_contrib = float(b_U[target_token])

    # Embed contribution
    embed_key = "hook_embed"
    pos_embed_key = "hook_pos_embed"
    embed_contrib = 0.0
    if embed_key in cache:
        embed_contrib += float(jnp.dot(cache[embed_key][position], unembed_dir))
    if pos_embed_key in cache:
        embed_contrib += float(jnp.dot(cache[pos_embed_key][position], unembed_dir))

    components = [{"name": "embed", "logit_contribution": embed_contrib}]

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        attn_contrib = 0.0
        mlp_contrib = 0.0
        if attn_key in cache:
            attn_contrib = float(jnp.dot(cache[attn_key][position], unembed_dir))
        if mlp_key in cache:
            mlp_contrib = float(jnp.dot(cache[mlp_key][position], unembed_dir))

        components.append({"name": f"L{layer}_attn", "logit_contribution": attn_contrib})
        components.append({"name": f"L{layer}_mlp", "logit_contribution": mlp_contrib})

    components.append({"name": "bias", "logit_contribution": bias_contrib})

    total = sum(c["logit_contribution"] for c in components)

    return {
        "components": components,
        "target_token": target_token,
        "total_logit": total,
        "position": position,
    }


def per_head_logit_attribution(model, tokens, position=-1, target_token=None):
    """Attribute the target logit to each individual attention head.

    Provides head-level granularity beyond layer-level decomposition.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    if target_token is None:
        final_key = f"blocks.{n_layers - 1}.hook_resid_post"
        if final_key not in cache:
            final_key = f"blocks.{n_layers - 1}.hook_resid_pre"
        logits = cache[final_key][position] @ W_U + b_U
        target_token = int(jnp.argmax(logits))

    unembed_dir = W_U[:, target_token]

    heads = []
    for layer in range(n_layers):
        z_key = f"blocks.{layer}.attn.hook_z"
        W_O = model.blocks[layer].attn.W_O

        if z_key not in cache:
            continue
        z = cache[z_key]  # [seq, n_heads, d_head]

        for h in range(n_heads):
            out = z[position, h] @ W_O[h]  # [d_model]
            contrib = float(jnp.dot(out, unembed_dir))
            heads.append({
                "layer": layer,
                "head": h,
                "logit_contribution": contrib,
                "output_norm": float(jnp.linalg.norm(out)),
            })

    heads.sort(key=lambda x: abs(x["logit_contribution"]), reverse=True)

    return {
        "heads": heads,
        "target_token": target_token,
        "position": position,
        "top_positive": [h for h in heads if h["logit_contribution"] > 0][:3],
        "top_negative": [h for h in heads if h["logit_contribution"] < 0][:3],
    }


def logit_attribution_by_position(model, tokens, target_position=-1):
    """Attribute the target logit to information from each source position.

    Combines attention patterns with value-through-OV-circuit to attribute
    the logit to specific source tokens.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)
    if target_position < 0:
        target_position = seq_len + target_position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Get target token
    final_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if final_key not in cache:
        final_key = f"blocks.{n_layers - 1}.hook_resid_pre"
    logits = cache[final_key][target_position] @ W_U + b_U
    target_token = int(jnp.argmax(logits))
    unembed_dir = W_U[:, target_token]

    per_source = jnp.zeros(seq_len)
    for layer in range(n_layers):
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        z_key = f"blocks.{layer}.attn.hook_z"
        W_O = model.blocks[layer].attn.W_O

        if pattern_key not in cache or z_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq, seq]
        v_key = f"blocks.{layer}.attn.hook_v"
        if v_key not in cache:
            continue
        v = cache[v_key]  # [seq, n_heads, d_head]

        for h in range(n_heads):
            for src in range(seq_len):
                attn_weight = float(pattern[h, target_position, src])
                # Value from source through OV
                val_out = v[src, h] @ W_O[h]
                logit_contrib = attn_weight * float(jnp.dot(val_out, unembed_dir))
                per_source = per_source.at[src].add(logit_contrib)

    source_results = []
    for src in range(seq_len):
        source_results.append({
            "position": src,
            "token_id": int(tokens[src]),
            "logit_contribution": float(per_source[src]),
        })

    source_results.sort(key=lambda x: abs(x["logit_contribution"]), reverse=True)

    return {
        "per_source": source_results,
        "target_token": target_token,
        "target_position": target_position,
    }


def top_promoted_suppressed(model, tokens, position=-1, top_k=10):
    """Find the most promoted and most suppressed tokens at a position.

    Returns the top-k tokens with highest and lowest logits.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]
    if position < 0:
        position = seq_len + position

    pos_logits = logits[position]  # [vocab]
    probs = jax.nn.softmax(pos_logits)

    top_indices = jnp.argsort(pos_logits)[-top_k:][::-1]
    bot_indices = jnp.argsort(pos_logits)[:top_k]

    promoted = []
    for idx in top_indices:
        promoted.append({
            "token_id": int(idx),
            "logit": float(pos_logits[int(idx)]),
            "probability": float(probs[int(idx)]),
        })

    suppressed = []
    for idx in bot_indices:
        suppressed.append({
            "token_id": int(idx),
            "logit": float(pos_logits[int(idx)]),
            "probability": float(probs[int(idx)]),
        })

    return {
        "promoted": promoted,
        "suppressed": suppressed,
        "position": position,
        "logit_range": float(jnp.max(pos_logits) - jnp.min(pos_logits)),
    }


def logit_attribution_summary(model, tokens, position=-1):
    """Summary of logit attribution across all components.

    Combines full decomposition with per-head attribution.
    """
    decomp = full_logit_decomposition(model, tokens, position=position)
    heads = per_head_logit_attribution(model, tokens, position=position,
                                        target_token=decomp["target_token"])

    # Aggregate
    total_attn = sum(c["logit_contribution"] for c in decomp["components"]
                     if "attn" in c["name"])
    total_mlp = sum(c["logit_contribution"] for c in decomp["components"]
                    if "mlp" in c["name"])

    return {
        "target_token": decomp["target_token"],
        "total_logit": decomp["total_logit"],
        "total_attn_contribution": total_attn,
        "total_mlp_contribution": total_mlp,
        "embed_contribution": decomp["components"][0]["logit_contribution"],
        "n_heads": len(heads["heads"]),
        "top_head": f"L{heads['heads'][0]['layer']}H{heads['heads'][0]['head']}" if heads["heads"] else "none",
        "top_head_contribution": heads["heads"][0]["logit_contribution"] if heads["heads"] else 0.0,
    }
