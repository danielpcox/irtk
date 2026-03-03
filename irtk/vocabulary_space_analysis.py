"""Vocabulary space analysis: analyze projections into vocabulary space.

Examine how representations map to tokens, token neighborhoods in
logit space, prediction diversity, and vocabulary coverage patterns.
"""

import jax
import jax.numpy as jnp


def logit_space_neighbors(model, tokens, position=-1, layer=-1, top_k=10):
    """Find the nearest neighbors in logit space at a given layer/position.

    Projects the residual stream through W_U to find promoted tokens.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position
    if layer < 0:
        layer = n_layers + layer

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache:
        key = f"blocks.{layer}.hook_resid_pre"
    rep = cache[key][position]  # [d_model]

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    logits = rep @ W_U + b_U  # [vocab]
    probs = jax.nn.softmax(logits)

    top_indices = jnp.argsort(logits)[-top_k:][::-1]
    neighbors = []
    for idx in top_indices:
        neighbors.append({
            "token_id": int(idx),
            "logit": float(logits[int(idx)]),
            "probability": float(probs[int(idx)]),
        })

    return {
        "neighbors": neighbors,
        "position": position,
        "layer": layer,
        "top_token": int(top_indices[0]),
        "top_prob": float(probs[int(top_indices[0])]),
    }


def vocabulary_coverage(model, tokens, layer=-1, threshold=0.01):
    """Analyze how many vocabulary tokens are assigned non-negligible probability.

    Measures the effective vocabulary size at a given layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if layer < 0:
        layer = n_layers + layer
    seq_len = len(tokens)

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_position = []
    for pos in range(seq_len):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][pos]
        logits = rep @ W_U + b_U
        probs = jax.nn.softmax(logits)

        n_above = int(jnp.sum(probs > threshold))
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
        max_prob = float(jnp.max(probs))

        per_position.append({
            "position": pos,
            "token_id": int(tokens[pos]),
            "n_tokens_above_threshold": n_above,
            "entropy": entropy,
            "max_prob": max_prob,
        })

    mean_coverage = sum(p["n_tokens_above_threshold"] for p in per_position) / max(len(per_position), 1)

    return {
        "per_position": per_position,
        "layer": layer,
        "mean_coverage": mean_coverage,
        "threshold": threshold,
    }


def prediction_diversity_across_positions(model, tokens, layer=-1):
    """Measure how diverse predictions are across different positions.

    If all positions predict the same token, diversity is low.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if layer < 0:
        layer = n_layers + layer
    seq_len = len(tokens)

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    top_tokens = []
    logit_vecs = []
    for pos in range(seq_len):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][pos]
        logits = rep @ W_U + b_U
        top_tokens.append(int(jnp.argmax(logits)))
        logit_vecs.append(logits)

    n_unique = len(set(top_tokens))

    # Pairwise logit cosine
    if len(logit_vecs) > 1:
        pairs = []
        for i in range(min(seq_len, 5)):
            for j in range(i + 1, min(seq_len, 5)):
                a = logit_vecs[i]
                b = logit_vecs[j]
                cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10))
                pairs.append(cos)
        mean_logit_sim = sum(pairs) / len(pairs)
    else:
        mean_logit_sim = 1.0

    return {
        "top_tokens": top_tokens,
        "n_unique_predictions": n_unique,
        "mean_logit_similarity": mean_logit_sim,
        "layer": layer,
        "is_diverse": n_unique > seq_len * 0.5,
    }


def token_logit_trajectory(model, tokens, target_token, position=-1):
    """Track a specific token's logit across all layers.

    Shows how a target token is promoted or suppressed through the network.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key][position]
        logits = rep @ W_U + b_U
        probs = jax.nn.softmax(logits)

        per_layer.append({
            "layer": layer,
            "target_logit": float(logits[target_token]),
            "target_prob": float(probs[target_token]),
            "target_rank": int(jnp.sum(logits > logits[target_token])),
        })

    return {
        "per_layer": per_layer,
        "target_token": target_token,
        "position": position,
        "final_rank": per_layer[-1]["target_rank"] if per_layer else -1,
    }


def vocabulary_space_summary(model, tokens):
    """Cross-layer summary of vocabulary space projections.

    Tracks prediction sharpness, diversity, and confidence at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"

        entropies = []
        max_probs = []
        top_toks = []
        for pos in range(seq_len):
            rep = cache[key][pos]
            logits = rep @ W_U + b_U
            probs = jax.nn.softmax(logits)
            entropies.append(float(-jnp.sum(probs * jnp.log(probs + 1e-10))))
            max_probs.append(float(jnp.max(probs)))
            top_toks.append(int(jnp.argmax(logits)))

        per_layer.append({
            "layer": layer,
            "mean_entropy": sum(entropies) / len(entropies),
            "mean_max_prob": sum(max_probs) / len(max_probs),
            "n_unique_predictions": len(set(top_toks)),
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
        "sharpening": per_layer[-1]["mean_entropy"] < per_layer[0]["mean_entropy"] if per_layer else False,
    }
