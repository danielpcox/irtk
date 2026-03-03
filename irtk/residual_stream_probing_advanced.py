"""Residual stream probing (advanced): sophisticated probing of residual representations.

Beyond basic linear probes — analyze residual stream for token identity recovery,
next-token prediction quality, positional information, and feature separability.
"""

import jax
import jax.numpy as jnp


def token_identity_recovery(model, tokens, layer=0):
    """Can we recover which token is at each position from the residual stream?

    Uses nearest-neighbor in embedding space to check identity recovery.

    Returns:
        dict with 'accuracy', 'per_position' recovery results.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    W_E = model.embed.W_E  # [d_vocab, d_model]

    seq_len = resid.shape[0]
    per_position = []
    correct = 0
    for pos in range(seq_len):
        h = resid[pos]  # [d_model]
        # Cosine similarity to all embeddings
        sims = h @ W_E.T / (jnp.linalg.norm(h) * jnp.linalg.norm(W_E, axis=-1) + 1e-10)
        predicted = int(jnp.argmax(sims))
        actual = int(tokens[pos])
        is_correct = predicted == actual
        if is_correct:
            correct += 1
        per_position.append({
            "position": pos,
            "actual_token": actual,
            "predicted_token": predicted,
            "correct": is_correct,
            "similarity": float(sims[actual]),
        })
    return {
        "accuracy": correct / seq_len,
        "per_position": per_position,
    }


def next_token_prediction_quality(model, tokens, layer=0):
    """How good is the residual stream at predicting the next token at this layer?

    Projects through unembedding to get logit-lens predictions.

    Returns:
        dict with 'accuracy', 'mean_rank', 'per_position'.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    seq_len = resid.shape[0]
    per_position = []
    correct = 0
    ranks = []
    for pos in range(seq_len - 1):
        logits = resid[pos] @ W_U + b_U
        pred = int(jnp.argmax(logits))
        actual_next = int(tokens[pos + 1])
        is_correct = pred == actual_next
        if is_correct:
            correct += 1
        # Rank of actual next token
        sorted_idx = jnp.argsort(-logits)
        rank = int(jnp.argmax(sorted_idx == actual_next))
        ranks.append(rank)
        per_position.append({
            "position": pos,
            "predicted": pred,
            "actual_next": actual_next,
            "correct": is_correct,
            "actual_rank": rank,
        })
    accuracy = correct / (seq_len - 1) if seq_len > 1 else 0.0
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    return {
        "accuracy": accuracy,
        "mean_rank": mean_rank,
        "per_position": per_position,
    }


def positional_information_content(model, tokens, layer=0):
    """How much positional vs content information is in the residual stream?

    Compare similarity between same-token-different-position pairs.

    Returns:
        dict with 'positional_score', 'content_score', 'dominant'.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]
    seq_len = resid.shape[0]

    # Content score: same token at different positions should be similar
    content_sims = []
    position_sims = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            cos = float(jnp.dot(resid[i], resid[j]) / (
                jnp.linalg.norm(resid[i]) * jnp.linalg.norm(resid[j]) + 1e-10
            ))
            if int(tokens[i]) == int(tokens[j]):
                content_sims.append(cos)
            else:
                position_sims.append(cos)

    content_score = sum(content_sims) / len(content_sims) if content_sims else 0.0
    position_score = sum(position_sims) / len(position_sims) if position_sims else 0.0

    dominant = "content" if content_score > position_score + 0.1 else (
        "position" if position_score > content_score + 0.1 else "mixed"
    )
    return {
        "content_score": content_score,
        "positional_score": position_score,
        "dominant": dominant,
        "n_same_token_pairs": len(content_sims),
        "n_diff_token_pairs": len(position_sims),
    }


def residual_feature_separability(model, tokens, layer=0):
    """How separable are different token representations in residual space?

    Uses ratio of between-class to within-class variance (Fisher criterion).

    Returns:
        dict with 'separability_score', 'n_unique_tokens'.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[("resid_post", layer)]  # [seq, d_model]

    # Group positions by token
    token_groups = {}
    for pos in range(len(tokens)):
        tok = int(tokens[pos])
        if tok not in token_groups:
            token_groups[tok] = []
        token_groups[tok].append(pos)

    if len(token_groups) < 2:
        return {"separability_score": 0.0, "n_unique_tokens": len(token_groups)}

    # Global mean
    global_mean = jnp.mean(resid, axis=0)

    # Between-class variance
    between_var = 0.0
    within_var = 0.0
    for tok, positions in token_groups.items():
        group_resids = jnp.stack([resid[p] for p in positions])
        group_mean = jnp.mean(group_resids, axis=0)
        between_var += len(positions) * float(jnp.sum((group_mean - global_mean) ** 2))
        for p in positions:
            within_var += float(jnp.sum((resid[p] - group_mean) ** 2))

    separability = between_var / (within_var + 1e-10)
    return {
        "separability_score": separability,
        "n_unique_tokens": len(token_groups),
        "between_variance": between_var,
        "within_variance": within_var,
    }


def residual_probing_summary(model, tokens):
    """Summary of residual stream probing across all layers.

    Returns:
        dict with 'per_layer' probing quality metrics.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        identity = token_identity_recovery(model, tokens, layer=layer)
        prediction = next_token_prediction_quality(model, tokens, layer=layer)
        per_layer.append({
            "layer": layer,
            "identity_accuracy": identity["accuracy"],
            "prediction_accuracy": prediction["accuracy"],
            "mean_prediction_rank": prediction["mean_rank"],
        })
    return {"per_layer": per_layer}
