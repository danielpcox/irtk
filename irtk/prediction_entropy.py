"""Prediction entropy and information budget analysis.

Analyze where in a sequence and at which layer the model is uncertain or
confident. Track entropy of the predicted distribution, per-token information
gain, surprisal profiles, and how early predictions become final.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_prediction_entropy(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    apply_ln: bool = True,
) -> np.ndarray:
    """Compute prediction entropy at every layer and position.

    Applies the unembedding to each layer's residual stream (logit lens style)
    and computes the entropy of the resulting distribution.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        apply_ln: Apply final layer norm before unembedding.

    Returns:
        [n_layers+1, seq_len] entropy values. Index 0 = after embedding,
        index i+1 = after layer i.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()  # [n_components, seq_len, d_model]

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    n_components = resid_stack.shape[0]
    entropies = []

    for i in range(n_components):
        resid = resid_stack[i]
        if apply_ln and model.ln_final is not None:
            resid = model.ln_final(resid)
        logits = resid @ W_U + b_U  # [seq_len, d_vocab]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)  # [seq_len]
        entropies.append(np.array(entropy))

    return np.stack(entropies, axis=0)


def prediction_commit_depth(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
    threshold: float = 0.5,
    apply_ln: bool = True,
) -> dict:
    """Find the earliest layer where the model commits to a prediction.

    Identifies the first layer at which the model's top-1 prediction matches
    target_token and the probability exceeds threshold.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        target_token: The token to check for.
        pos: Sequence position to inspect (-1 for last).
        threshold: Probability threshold for "commit".
        apply_ln: Apply final layer norm.

    Returns:
        Dict with:
        - "commit_layer": earliest layer meeting criteria (None if never)
        - "probability_trajectory": [n_layers+1] probability of target_token
        - "top1_trajectory": [n_layers+1] top-1 token IDs at each layer
        - "final_probability": probability at the final layer
    """
    tokens = jnp.array(tokens)
    resolved_pos = pos if pos >= 0 else len(tokens) + pos

    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    n_components = resid_stack.shape[0]
    prob_traj = []
    top1_traj = []

    for i in range(n_components):
        resid = resid_stack[i]
        if apply_ln and model.ln_final is not None:
            resid = model.ln_final(resid)
        logits = resid @ W_U + b_U
        probs = jax.nn.softmax(logits, axis=-1)
        pos_probs = probs[resolved_pos]
        prob_traj.append(float(pos_probs[target_token]))
        top1_traj.append(int(jnp.argmax(pos_probs)))

    # Find commit layer
    commit_layer = None
    for i in range(n_components):
        if top1_traj[i] == target_token and prob_traj[i] >= threshold:
            commit_layer = i
            break

    return {
        "commit_layer": commit_layer,
        "probability_trajectory": np.array(prob_traj),
        "top1_trajectory": np.array(top1_traj),
        "final_probability": prob_traj[-1],
    }


def per_token_surprisal(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Compute surprisal at each token position.

    Surprisal = -log P(token[t] | tokens[:t]) using the model's actual
    output distribution.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.

    Returns:
        Dict with:
        - "surprisals": [seq_len-1] surprisal for each next-token prediction
        - "mean_surprisal": average surprisal (= cross-entropy loss in nats)
        - "most_surprising_pos": position with highest surprisal
        - "least_surprising_pos": position with lowest surprisal
    """
    tokens = jnp.array(tokens)
    logits = model(tokens)  # [seq_len, d_vocab]

    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [seq_len, d_vocab]

    # Surprisal of next token at each position
    # Position t predicts tokens[t+1]
    target_tokens = tokens[1:]  # [seq_len-1]
    surprisals = []
    for t in range(len(target_tokens)):
        s = -float(log_probs[t, int(target_tokens[t])])
        surprisals.append(s)

    surprisals = np.array(surprisals)
    mean_surprisal = float(np.mean(surprisals))

    return {
        "surprisals": surprisals,
        "mean_surprisal": mean_surprisal,
        "most_surprising_pos": int(np.argmax(surprisals)),
        "least_surprising_pos": int(np.argmin(surprisals)),
    }


def entropy_reduction_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    apply_ln: bool = True,
) -> dict:
    """Measure how much each layer reduces prediction entropy.

    delta_H[layer] = H(layer-1) - H(layer). Positive = entropy-reducing
    (converging on prediction), negative = entropy-increasing.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze (-1 for last).
        apply_ln: Apply final layer norm.

    Returns:
        Dict with:
        - "entropy_per_layer": [n_layers+1] entropy at each layer
        - "delta_entropy": [n_layers] entropy reduction per layer
        - "biggest_reducer": layer index with largest entropy reduction
        - "total_reduction": total entropy reduction from embedding to final
    """
    entropies = layer_prediction_entropy(model, tokens, apply_ln=apply_ln)
    resolved_pos = pos if pos >= 0 else entropies.shape[1] + pos

    layer_entropies = entropies[:, resolved_pos]  # [n_layers+1]
    deltas = layer_entropies[:-1] - layer_entropies[1:]  # [n_layers]

    return {
        "entropy_per_layer": layer_entropies,
        "delta_entropy": deltas,
        "biggest_reducer": int(np.argmax(deltas)),
        "total_reduction": float(layer_entropies[0] - layer_entropies[-1]),
    }


def compare_entropy_profiles(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
    pos: int = -1,
    apply_ln: bool = True,
) -> dict:
    """Compare layer-by-layer entropy profiles for two inputs.

    Useful for contrastive analysis: "where does the model first treat
    these two prompts differently?"

    Args:
        model: HookedTransformer.
        tokens_a: First token sequence.
        tokens_b: Second token sequence.
        pos: Position to compare (-1 for last).
        apply_ln: Apply final layer norm.

    Returns:
        Dict with:
        - "entropy_a": [n_layers+1] entropy for first input
        - "entropy_b": [n_layers+1] entropy for second input
        - "absolute_diff": [n_layers+1] |H_a - H_b| per layer
        - "divergence_layer": first layer where profiles diverge significantly
        - "max_diff_layer": layer with largest absolute difference
    """
    ent_a = layer_prediction_entropy(model, tokens_a, apply_ln=apply_ln)
    ent_b = layer_prediction_entropy(model, tokens_b, apply_ln=apply_ln)

    resolved_pos_a = pos if pos >= 0 else ent_a.shape[1] + pos
    resolved_pos_b = pos if pos >= 0 else ent_b.shape[1] + pos

    profile_a = ent_a[:, resolved_pos_a]
    profile_b = ent_b[:, resolved_pos_b]
    abs_diff = np.abs(profile_a - profile_b)

    # Find divergence layer: first layer where diff exceeds 10% of max entropy
    max_ent = max(np.max(profile_a), np.max(profile_b), 1e-10)
    divergence_threshold = 0.1 * max_ent
    divergence_layer = None
    for i in range(len(abs_diff)):
        if abs_diff[i] > divergence_threshold:
            divergence_layer = i
            break

    return {
        "entropy_a": profile_a,
        "entropy_b": profile_b,
        "absolute_diff": abs_diff,
        "divergence_layer": divergence_layer,
        "max_diff_layer": int(np.argmax(abs_diff)),
    }
