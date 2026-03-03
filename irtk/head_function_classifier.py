"""Head function classifier: automatically classify attention head roles.

Classify heads into functional categories (previous token, induction,
positional, copying, inhibition) based on attention pattern analysis.
"""

import jax
import jax.numpy as jnp


def classify_previous_token(model, tokens, layer=0, threshold=0.3):
    """Score each head on previous-token behavior.

    Previous-token heads attend primarily to position i-1 from position i.

    Returns:
        dict with 'per_head' list of scores and classifications.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        # Diagonal offset by -1: position i attends to i-1
        score = 0.0
        count = 0
        for i in range(1, seq_len):
            score += float(pat[i, i - 1])
            count += 1
        avg_score = score / count if count > 0 else 0.0
        per_head.append({
            "head": int(h),
            "previous_token_score": avg_score,
            "is_previous_token": avg_score > threshold,
        })
    return {"per_head": per_head}


def classify_induction(model, tokens, layer=0, threshold=0.2):
    """Score each head on induction behavior.

    Induction heads attend to positions where the preceding token
    matches the current query's preceding token.

    Returns:
        dict with 'per_head' list of induction scores.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]
        induction_attn = 0.0
        count = 0
        for q in range(2, seq_len):
            for k in range(1, q):
                if int(tokens[k - 1]) == int(tokens[q - 1]):
                    induction_attn += float(pat[q, k])
                    count += 1
        avg = induction_attn / count if count > 0 else 0.0
        per_head.append({
            "head": int(h),
            "induction_score": avg,
            "is_induction": avg > threshold,
        })
    return {"per_head": per_head}


def classify_copying(model, tokens, layer=0, top_k=3):
    """Score each head on copying behavior.

    Copying heads: the OV circuit maps source token embeddings to
    the same token in the output. Measured via W_E @ W_V @ W_O @ W_U.

    Returns:
        dict with 'per_head' scores and top copied tokens.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    n_heads = W_V.shape[0]
    d_vocab = W_E.shape[0]
    per_head = []
    for h in range(n_heads):
        # OV circuit for this head: [d_model, d_model]
        ov = W_V[h] @ W_O[h]  # [d_model, d_model]
        # Full circuit: W_E @ OV @ W_U -> [d_vocab, d_vocab]
        full = W_E @ ov @ W_U  # [d_vocab, d_vocab]
        # Copying score: mean of diagonal (how much each token maps to itself)
        diag = jnp.diag(full)  # [d_vocab]
        copy_score = float(jnp.mean(diag))
        # Mean of off-diagonal for comparison
        off_diag = float((jnp.sum(full) - jnp.sum(diag)) / (d_vocab * (d_vocab - 1)))
        # Top copied tokens
        top_idx = jnp.argsort(-diag)[:top_k]
        top_copied = [(int(idx), float(diag[idx])) for idx in top_idx]
        per_head.append({
            "head": int(h),
            "copy_score": copy_score,
            "off_diagonal_mean": off_diag,
            "copy_advantage": copy_score - off_diag,
            "top_copied_tokens": top_copied,
        })
    return {"per_head": per_head}


def classify_positional(model, tokens, layer=0, threshold=0.3):
    """Score each head on positional (uniform/fixed pattern) behavior.

    Positional heads have low variance in their attention patterns
    across different query positions.

    Returns:
        dict with 'per_head' scores.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        # Variance across query positions for each key position
        var_across_queries = jnp.var(pat, axis=0)  # [seq]
        mean_var = float(jnp.mean(var_across_queries))
        # Low variance = positional (same pattern regardless of content)
        positional_score = 1.0 / (1.0 + mean_var * 100)
        per_head.append({
            "head": int(h),
            "positional_score": positional_score,
            "mean_pattern_variance": mean_var,
            "is_positional": positional_score > threshold,
        })
    return {"per_head": per_head}


def head_function_summary(model, tokens):
    """Classify all heads across all layers.

    Returns:
        dict with 'per_layer' list, each containing per-head classifications
        and layer-level counts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        prev = classify_previous_token(model, tokens, layer=layer)
        ind = classify_induction(model, tokens, layer=layer)
        pos = classify_positional(model, tokens, layer=layer)
        n_heads = len(prev["per_head"])
        heads = []
        for h in range(n_heads):
            roles = []
            if prev["per_head"][h]["is_previous_token"]:
                roles.append("previous_token")
            if ind["per_head"][h]["is_induction"]:
                roles.append("induction")
            if pos["per_head"][h]["is_positional"]:
                roles.append("positional")
            if not roles:
                roles.append("unclassified")
            heads.append({
                "head": h,
                "roles": roles,
                "previous_token_score": prev["per_head"][h]["previous_token_score"],
                "induction_score": ind["per_head"][h]["induction_score"],
                "positional_score": pos["per_head"][h]["positional_score"],
            })
        per_layer.append({
            "layer": layer,
            "per_head": heads,
            "n_previous_token": sum(1 for hd in heads if "previous_token" in hd["roles"]),
            "n_induction": sum(1 for hd in heads if "induction" in hd["roles"]),
            "n_positional": sum(1 for hd in heads if "positional" in hd["roles"]),
        })
    return {"per_layer": per_layer}
