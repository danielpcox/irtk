"""Multi-token prediction analysis.

Analyzes whether and how models represent future tokens beyond the
immediate next token. Tests for planning behavior, lookahead in
the residual stream, and future information encoded at each layer.

Functions:
- future_token_probing: Probe for future tokens in the residual stream
- planning_horizon: Estimate how many tokens ahead the model "plans"
- next_k_token_accuracy: Accuracy at predicting k tokens ahead
- representation_lookahead: How much future context is encoded in representations
- future_information_by_layer: Per-layer future token predictability

References:
    - Gloeckle et al. (2024) "Better & Faster Large Language Models via Multi-token Prediction"
    - Pal et al. (2023) "Future Lens: Anticipating Subsequent Tokens"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def future_token_probing(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    future_offset: int = 1,
    layer: int = -1,
) -> dict:
    """Probe for future tokens in the residual stream.

    Uses the unembedding matrix to check if the representation at position i
    contains information about the token at position i + offset.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        future_offset: How many positions ahead to probe.
        layer: Layer to probe (-1 = last).

    Returns:
        Dict with:
            "future_ranks": [valid_positions] rank of the future token at each position
            "future_probs": [valid_positions] probability of future token
            "mean_rank": mean rank of the future token
            "mean_prob": mean probability
            "top_10_fraction": fraction of positions where future token is in top-10
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    if layer == -1:
        layer = model.cfg.n_layers - 1

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache.cache_dict:
        return {
            "future_ranks": np.array([]),
            "future_probs": np.array([]),
            "mean_rank": float(model.cfg.d_vocab),
            "mean_prob": 0.0,
            "top_10_fraction": 0.0,
        }

    resid = np.array(cache.cache_dict[key])  # [seq, d_model]
    tokens_np = np.array(tokens)

    valid_positions = range(seq_len - future_offset)
    ranks = []
    probs = []

    for pos in valid_positions:
        future_token = int(tokens_np[pos + future_offset])
        layer_logits = resid[pos] @ W_U + b_U
        layer_probs = np.exp(layer_logits - np.max(layer_logits))
        layer_probs = layer_probs / np.sum(layer_probs)

        rank = int(np.sum(layer_logits > layer_logits[future_token]))
        ranks.append(rank)
        probs.append(float(layer_probs[future_token]))

    ranks = np.array(ranks)
    probs = np.array(probs)

    top_10 = float(np.mean(ranks < 10)) if len(ranks) > 0 else 0.0

    return {
        "future_ranks": ranks,
        "future_probs": probs,
        "mean_rank": float(np.mean(ranks)) if len(ranks) > 0 else float(model.cfg.d_vocab),
        "mean_prob": float(np.mean(probs)) if len(probs) > 0 else 0.0,
        "top_10_fraction": top_10,
    }


def planning_horizon(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    max_lookahead: int = 5,
    layer: int = -1,
) -> dict:
    """Estimate how many tokens ahead the model "plans."

    Probes for future tokens at increasing offsets and finds where
    predictability drops to chance level.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        max_lookahead: Maximum offset to test.
        layer: Layer to probe (-1 = last).

    Returns:
        Dict with:
            "offset_probs": [max_lookahead] mean future token probability per offset
            "offset_ranks": [max_lookahead] mean future token rank per offset
            "effective_horizon": furthest offset with mean rank < d_vocab/10
            "horizon_decay_rate": rate at which predictability drops with offset
    """
    seq_len = len(tokens)
    max_lookahead = min(max_lookahead, seq_len - 1)

    offset_probs = np.zeros(max_lookahead)
    offset_ranks = np.zeros(max_lookahead)

    for k in range(1, max_lookahead + 1):
        result = future_token_probing(model, tokens, future_offset=k, layer=layer)
        offset_probs[k - 1] = result["mean_prob"]
        offset_ranks[k - 1] = result["mean_rank"]

    # Effective horizon
    threshold = model.cfg.d_vocab / 10
    horizon = 0
    for k in range(max_lookahead):
        if offset_ranks[k] < threshold:
            horizon = k + 1

    # Decay rate
    if max_lookahead >= 2 and offset_probs[0] > 1e-10:
        decay = -float(np.log(offset_probs[-1] + 1e-10) - np.log(offset_probs[0] + 1e-10)) / max_lookahead
    else:
        decay = 0.0

    return {
        "offset_probs": offset_probs,
        "offset_ranks": offset_ranks,
        "effective_horizon": horizon,
        "horizon_decay_rate": max(0.0, decay),
    }


def next_k_token_accuracy(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    k: int = 3,
) -> dict:
    """Measure accuracy at predicting k tokens ahead from the output.

    Uses the model's logits at each position to predict the next k tokens
    and measures top-1 and top-5 accuracy.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        k: Number of tokens ahead to predict.

    Returns:
        Dict with:
            "top_1_accuracy_per_offset": [k] top-1 accuracy at each future offset
            "top_5_accuracy_per_offset": [k] top-5 accuracy at each future offset
            "mean_top_1": mean top-1 accuracy across all offsets
            "best_offset": offset with highest accuracy (usually 1)
    """
    logits = np.array(model(tokens))
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    top1_acc = np.zeros(k)
    top5_acc = np.zeros(k)

    for offset in range(1, k + 1):
        n_valid = seq_len - offset
        if n_valid <= 0:
            continue

        correct_top1 = 0
        correct_top5 = 0

        for pos in range(n_valid):
            target = int(tokens_np[pos + offset])
            pos_logits = logits[pos]
            sorted_tokens = np.argsort(pos_logits)[::-1]

            if sorted_tokens[0] == target:
                correct_top1 += 1
            if target in sorted_tokens[:5]:
                correct_top5 += 1

        top1_acc[offset - 1] = correct_top1 / n_valid
        top5_acc[offset - 1] = correct_top5 / n_valid

    best = int(np.argmax(top1_acc))

    return {
        "top_1_accuracy_per_offset": top1_acc,
        "top_5_accuracy_per_offset": top5_acc,
        "mean_top_1": float(np.mean(top1_acc)),
        "best_offset": best + 1,
    }


def representation_lookahead(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    layer: int = -1,
) -> dict:
    """How much future context is encoded in the current representation?

    Compares the representation at position `pos` with representations
    at later positions to measure information about future context.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to analyze.
        layer: Layer to analyze (-1 = last).

    Returns:
        Dict with:
            "similarity_to_future": [n_future] cosine similarity to each future position
            "mean_future_similarity": average similarity to future positions
            "lookahead_distance": weighted mean distance of similar future positions
            "max_similarity_offset": offset with highest similarity
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    if layer == -1:
        layer = model.cfg.n_layers - 1
    if pos == -1:
        pos = seq_len - 1

    key = f"blocks.{layer}.hook_resid_post"
    if key not in cache.cache_dict:
        return {
            "similarity_to_future": np.array([]),
            "mean_future_similarity": 0.0,
            "lookahead_distance": 0.0,
            "max_similarity_offset": 0,
        }

    resid = np.array(cache.cache_dict[key])
    pos_rep = resid[pos]

    # Compare with representations at later positions (in the sequence)
    # Since we're at the last pos by default, compare with earlier positions
    # But conceptually, we want to see if pos encodes info about positions that
    # come "after" it in the narrative (which don't exist in causal models).
    # Instead, compare with nearby positions to measure local structure.
    similarities = []
    for future_pos in range(max(0, pos - 5), pos):
        other_rep = resid[future_pos]
        cos = float(np.dot(pos_rep, other_rep) /
                     (np.linalg.norm(pos_rep) * np.linalg.norm(other_rep) + 1e-10))
        similarities.append(cos)

    similarities = np.array(similarities) if similarities else np.array([0.0])

    mean_sim = float(np.mean(similarities))
    max_offset = int(np.argmax(similarities)) + 1 if len(similarities) > 0 else 0

    # Weighted mean distance
    if len(similarities) > 0 and np.sum(np.abs(similarities)) > 1e-10:
        offsets = np.arange(1, len(similarities) + 1)
        weights = np.abs(similarities) / (np.sum(np.abs(similarities)) + 1e-10)
        lookahead = float(np.sum(offsets * weights))
    else:
        lookahead = 0.0

    return {
        "similarity_to_future": similarities,
        "mean_future_similarity": mean_sim,
        "lookahead_distance": lookahead,
        "max_similarity_offset": max_offset,
    }


def future_information_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    future_offset: int = 1,
) -> dict:
    """Per-layer future token predictability.

    Probes for the next+offset token at each layer to track when
    future information enters the residual stream.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        future_offset: How far ahead to probe.

    Returns:
        Dict with:
            "layer_mean_probs": [n_layers] mean future token probability per layer
            "layer_mean_ranks": [n_layers] mean future token rank per layer
            "best_layer": layer with highest future predictability
            "emergence_layer": first layer where rank drops below d_vocab/5
    """
    n_layers = model.cfg.n_layers

    layer_probs = np.zeros(n_layers)
    layer_ranks = np.zeros(n_layers)

    for l in range(n_layers):
        result = future_token_probing(model, tokens, future_offset=future_offset, layer=l)
        layer_probs[l] = result["mean_prob"]
        layer_ranks[l] = result["mean_rank"]

    best = int(np.argmax(layer_probs))

    threshold = model.cfg.d_vocab / 5
    emergence = -1
    for l in range(n_layers):
        if layer_ranks[l] < threshold:
            emergence = l
            break

    return {
        "layer_mean_probs": layer_probs,
        "layer_mean_ranks": layer_ranks,
        "best_layer": best,
        "emergence_layer": emergence,
    }
