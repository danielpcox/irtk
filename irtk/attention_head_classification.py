"""Attention head classification: classify heads by functional role.

Detect induction heads, previous-token heads, positional heads,
copy heads, and other behavioral archetypes from attention patterns.
"""

import jax
import jax.numpy as jnp


def detect_induction_heads(model, tokens):
    """Detect induction heads by measuring attention to token-after-previous-occurrence.

    An induction head attends from position i to position j where
    tokens[j-1] == tokens[i-1] (attending to what came after the
    previous occurrence of the current token's predecessor).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    results = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        for head in range(n_heads):
            attn = pattern[head]  # [seq, seq]
            # Build induction mask: position j is an induction target for query i
            # if tokens[j-1] == tokens[i-1]
            induction_score = 0.0
            count = 0
            for qi in range(2, seq_len):
                for ki in range(1, qi):
                    if int(tokens[qi - 1]) == int(tokens[ki - 1]):
                        induction_score += float(attn[qi, ki])
                        count += 1
            avg_score = induction_score / max(count, 1)
            results.append({
                "layer": layer,
                "head": head,
                "induction_score": avg_score,
                "match_count": count,
                "is_induction": avg_score > 0.3,
            })

    return {
        "heads": sorted(results, key=lambda x: x["induction_score"], reverse=True),
        "n_induction_heads": sum(1 for r in results if r["is_induction"]),
    }


def detect_previous_token_heads(model, tokens):
    """Detect heads that attend primarily to the immediately previous token.

    Measures diagonal-1 attention mass.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    results = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        for head in range(n_heads):
            attn = pattern[head]  # [seq, seq]
            # Measure attention on the -1 diagonal
            prev_mass = 0.0
            count = 0
            for i in range(1, seq_len):
                prev_mass += float(attn[i, i - 1])
                count += 1
            avg_prev = prev_mass / max(count, 1)
            results.append({
                "layer": layer,
                "head": head,
                "prev_token_score": avg_prev,
                "is_previous_token": avg_prev > 0.3,
            })

    return {
        "heads": sorted(results, key=lambda x: x["prev_token_score"], reverse=True),
        "n_previous_token_heads": sum(1 for r in results if r["is_previous_token"]),
    }


def detect_positional_heads(model, tokens):
    """Detect heads with position-dependent attention (fixed patterns).

    A positional head has low variance across different tokens but
    consistent patterns — measured by entropy of attention distribution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    results = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        if key not in cache:
            continue
        pattern = cache[key]  # [n_heads, seq, seq]
        for head in range(n_heads):
            attn = pattern[head]  # [seq, seq]
            # Measure BOS attention (first token)
            bos_mass = float(jnp.mean(attn[1:, 0])) if seq_len > 1 else 0.0
            # Measure self-attention (diagonal)
            self_mass = float(jnp.mean(jnp.diag(attn)))
            # Mean entropy
            entropy = -jnp.sum(attn * jnp.log(attn + 1e-10), axis=-1)
            mean_entropy = float(jnp.mean(entropy[1:])) if seq_len > 1 else 0.0
            max_entropy = float(jnp.log(jnp.arange(1, seq_len + 1).astype(jnp.float32)).mean())

            results.append({
                "layer": layer,
                "head": head,
                "bos_attention": bos_mass,
                "self_attention": self_mass,
                "mean_entropy": mean_entropy,
                "is_positional": bos_mass > 0.4 or self_mass > 0.4,
            })

    return {
        "heads": sorted(results, key=lambda x: max(x["bos_attention"], x["self_attention"]), reverse=True),
        "n_positional_heads": sum(1 for r in results if r["is_positional"]),
    }


def detect_copy_heads(model, tokens):
    """Detect heads that copy token identity (OV circuit projects through embeddings).

    A copy head's OV circuit W_V @ W_O maps token embeddings to similar output directions.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    W_E = model.embed.W_E  # [vocab, d_model]

    results = []
    for layer in range(n_layers):
        block = model.blocks[layer]
        W_V = block.attn.W_V  # [n_heads, d_model, d_head]
        W_O = block.attn.W_O  # [n_heads, d_head, d_model]
        for head in range(n_heads):
            # OV circuit: d_model -> d_head -> d_model
            OV = W_V[head] @ W_O[head]  # [d_model, d_model]
            # Apply OV to embeddings
            output = W_E @ OV  # [vocab, d_model]
            # Check if output aligns with input embeddings
            # Cosine between input and output for each token
            in_norms = jnp.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10
            out_norms = jnp.linalg.norm(output, axis=-1, keepdims=True) + 1e-10
            cos = jnp.sum((W_E / in_norms) * (output / out_norms), axis=-1)
            mean_copy_score = float(jnp.mean(cos))
            max_copy_score = float(jnp.max(cos))

            results.append({
                "layer": layer,
                "head": head,
                "mean_copy_score": mean_copy_score,
                "max_copy_score": max_copy_score,
                "is_copy": mean_copy_score > 0.3,
            })

    return {
        "heads": sorted(results, key=lambda x: x["mean_copy_score"], reverse=True),
        "n_copy_heads": sum(1 for r in results if r["is_copy"]),
    }


def head_classification_summary(model, tokens):
    """Classify all heads into functional categories.

    Categories: induction, previous_token, positional, copy, other.
    """
    induction = detect_induction_heads(model, tokens)
    prev_tok = detect_previous_token_heads(model, tokens)
    positional = detect_positional_heads(model, tokens)
    copy = detect_copy_heads(model, tokens)

    # Build lookup
    induction_set = {(h["layer"], h["head"]) for h in induction["heads"] if h["is_induction"]}
    prev_set = {(h["layer"], h["head"]) for h in prev_tok["heads"] if h["is_previous_token"]}
    pos_set = {(h["layer"], h["head"]) for h in positional["heads"] if h["is_positional"]}
    copy_set = {(h["layer"], h["head"]) for h in copy["heads"] if h["is_copy"]}

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    classifications = []
    for layer in range(n_layers):
        for head in range(n_heads):
            key = (layer, head)
            cats = []
            if key in induction_set:
                cats.append("induction")
            if key in prev_set:
                cats.append("previous_token")
            if key in pos_set:
                cats.append("positional")
            if key in copy_set:
                cats.append("copy")
            if not cats:
                cats.append("other")
            classifications.append({
                "layer": layer,
                "head": head,
                "categories": cats,
                "primary": cats[0],
            })

    return {
        "classifications": classifications,
        "n_induction": len(induction_set),
        "n_previous_token": len(prev_set),
        "n_positional": len(pos_set),
        "n_copy": len(copy_set),
        "n_other": sum(1 for c in classifications if c["primary"] == "other"),
    }
