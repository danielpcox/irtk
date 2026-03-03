"""Attention head specialization: classify heads by functional role.

Detect induction heads, previous-token heads, positional heads,
and other common attention head types.
"""

import jax.numpy as jnp


def previous_token_score(model, tokens, layer=0):
    """Score each head on how much it attends to the previous token.

    A previous-token head has high attention on the diagonal offset by 1.

    Returns:
        dict with 'per_head' list of scores, 'n_previous_token' count.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]  # [n_heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]
        score = 0.0
        count = 0
        for i in range(1, seq_len):
            score += float(pat[i, i - 1])
            count += 1
        avg_score = score / count if count > 0 else 0.0
        per_head.append({
            "head": int(h),
            "previous_token_score": avg_score,
            "is_previous_token": avg_score > 0.3,
        })
    n_prev = sum(1 for p in per_head if p["is_previous_token"])
    return {"per_head": per_head, "n_previous_token": n_prev}


def induction_score(model, tokens, layer=0):
    """Score each head on induction behavior.

    Induction heads attend to tokens that follow a copy of the current token.
    Approximated by checking if attention correlates with pattern repetition.

    Returns:
        dict with 'per_head' list, 'n_induction' count.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]
        score = 0.0
        count = 0
        for q in range(2, seq_len):
            for k in range(1, q):
                if int(tokens[k - 1]) == int(tokens[q - 1]):
                    score += float(pat[q, k])
                    count += 1
        avg = score / count if count > 0 else 0.0
        per_head.append({
            "head": int(h),
            "induction_score": avg,
            "is_induction": avg > 0.3,
        })
    n_ind = sum(1 for p in per_head if p["is_induction"])
    return {"per_head": per_head, "n_induction": n_ind}


def positional_head_score(model, tokens, layer=0):
    """Score each head on positional attention (consistent pattern across tokens).

    Uses variance of attention pattern across different query positions.

    Returns:
        dict with 'per_head' list, 'n_positional' count.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    n_heads = patterns.shape[0]
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]  # [seq, seq]
        var_across_queries = float(jnp.mean(jnp.var(pat, axis=0)))
        score = 1.0 / (1.0 + var_across_queries * 100)
        per_head.append({
            "head": int(h),
            "positional_score": score,
            "is_positional": score > 0.5,
        })
    n_pos = sum(1 for p in per_head if p["is_positional"])
    return {"per_head": per_head, "n_positional": n_pos}


def head_entropy_profile(model, tokens, layer=0):
    """Entropy profile: distinguish focused vs diffuse heads.

    Returns:
        dict with 'per_head' list with entropy and classification.
    """
    _, cache = model.run_with_cache(tokens)
    patterns = cache[("pattern", layer)]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]
    max_entropy = float(jnp.log(jnp.array(seq_len, dtype=jnp.float32)))
    per_head = []
    for h in range(n_heads):
        pat = patterns[h]
        entropy = float(-jnp.sum(pat * jnp.log(pat + 1e-10)) / seq_len)
        normalized = entropy / (max_entropy + 1e-10)
        classification = "focused" if normalized < 0.5 else "diffuse"
        per_head.append({
            "head": int(h),
            "entropy": entropy,
            "normalized_entropy": normalized,
            "classification": classification,
        })
    return {"per_head": per_head}


def head_specialization_summary(model, tokens):
    """Summary of head specialization across all layers.

    Returns:
        dict with 'per_layer' list of summary dicts.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        prev = previous_token_score(model, tokens, layer=layer)
        ind = induction_score(model, tokens, layer=layer)
        pos = positional_head_score(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "n_previous_token": prev["n_previous_token"],
            "n_induction": ind["n_induction"],
            "n_positional": pos["n_positional"],
        })
    return {"per_layer": per_layer}
