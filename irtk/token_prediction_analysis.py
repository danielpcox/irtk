"""Per-token prediction analysis.

Analyzes model predictions at each position in a sequence: confidence
distribution, surprise localization, token difficulty profiling, and
prediction agreement across layers.

References:
    Geva et al. (2022) "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts"
"""

import jax
import jax.numpy as jnp
import numpy as np


def per_token_confidence(model, tokens):
    """Compute model confidence for each token prediction in a sequence.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            top1_probs: array [seq_len] of probability of top predicted token
            top1_tokens: array [seq_len] of top predicted tokens
            entropies: array [seq_len] of prediction entropy per position
            mean_confidence: float
            least_confident_position: int
            most_confident_position: int
    """
    logits = model(tokens)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)

    top1_tokens = np.array(jnp.argmax(probs, axis=-1))
    top1_probs = np.array(jnp.max(probs, axis=-1))

    entropies = np.array(-jnp.sum(probs * log_probs, axis=-1))

    return {
        "top1_probs": top1_probs,
        "top1_tokens": top1_tokens,
        "entropies": entropies,
        "mean_confidence": float(np.mean(top1_probs)),
        "least_confident_position": int(np.argmin(top1_probs)),
        "most_confident_position": int(np.argmax(top1_probs)),
    }


def surprisal_profile(model, tokens):
    """Compute surprisal (negative log probability) for each actual next token.

    Positions where the model is "surprised" by the actual next token
    are where the model fails to predict correctly.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            surprisals: array [seq_len-1] of surprisal for each next token
            mean_surprisal: float
            max_surprisal_position: int
            min_surprisal_position: int
            correct_predictions: array [seq_len-1] of bool, whether top-1 was correct
            accuracy: float, fraction of correct predictions
    """
    logits = model(tokens)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    seq_len = len(tokens)
    surprisals = np.zeros(seq_len - 1)
    correct = np.zeros(seq_len - 1, dtype=bool)

    for i in range(seq_len - 1):
        next_token = int(tokens[i + 1])
        surprisals[i] = -float(log_probs[i, next_token])
        predicted = int(jnp.argmax(logits[i]))
        correct[i] = predicted == next_token

    return {
        "surprisals": surprisals,
        "mean_surprisal": float(np.mean(surprisals)),
        "max_surprisal_position": int(np.argmax(surprisals)),
        "min_surprisal_position": int(np.argmin(surprisals)),
        "correct_predictions": correct,
        "accuracy": float(np.mean(correct)),
    }


def token_difficulty_profile(model, tokens, n_samples=5, seed=42):
    """Profile token prediction difficulty across different inputs.

    Runs the model on multiple inputs and measures per-position statistics
    to identify positions that are consistently hard or easy to predict.

    Args:
        model: HookedTransformer model.
        tokens: Primary input token IDs [seq_len].
        n_samples: Number of random inputs for comparison.
        seed: Random seed.

    Returns:
        dict with:
            position_entropy_mean: array [seq_len] of mean entropy across inputs
            position_entropy_std: array [seq_len] of entropy variability
            hardest_position: int, position with highest mean entropy
            easiest_position: int, position with lowest mean entropy
            relative_difficulty: array [seq_len] of normalized difficulty scores
    """
    seq_len = len(tokens)

    # Primary input
    logits = model(tokens)
    probs = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    primary_ent = np.array(-jnp.sum(probs * log_probs, axis=-1))

    all_entropies = [primary_ent]

    rng = np.random.RandomState(seed)
    for _ in range(n_samples):
        rand_tokens = jnp.array(rng.randint(0, model.cfg.d_vocab, size=seq_len))
        logits_r = model(rand_tokens)
        probs_r = jnp.exp(jax.nn.log_softmax(logits_r, axis=-1))
        log_probs_r = jax.nn.log_softmax(logits_r, axis=-1)
        ent = np.array(-jnp.sum(probs_r * log_probs_r, axis=-1))
        all_entropies.append(ent)

    stacked = np.stack(all_entropies)
    means = np.mean(stacked, axis=0)
    stds = np.std(stacked, axis=0)

    max_ent = np.max(means)
    min_ent = np.min(means)
    if max_ent - min_ent > 1e-10:
        relative = (means - min_ent) / (max_ent - min_ent)
    else:
        relative = np.zeros(seq_len)

    return {
        "position_entropy_mean": means,
        "position_entropy_std": stds,
        "hardest_position": int(np.argmax(means)),
        "easiest_position": int(np.argmin(means)),
        "relative_difficulty": relative,
    }


def prediction_agreement_by_layer(model, tokens, pos=-1, top_k=5):
    """Measure agreement between predictions at different layers.

    Uses the logit lens to check whether different layers agree on the
    top-k predictions at a given position.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.
        top_k: Number of top tokens to compare.

    Returns:
        dict with:
            layer_top_k: array [n_layers+1, top_k] of top predicted tokens at each layer
            agreement_with_final: array [n_layers] of overlap fraction with final prediction
            first_agreement_layer: int, first layer that agrees with final on top-1
            consensus_fraction: float, fraction of layers agreeing on top-1
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    W_U = model.unembed.W_U
    b_U = getattr(model.unembed, 'b_U', None)

    final_top_k = np.array(jnp.argsort(logits[pos])[-top_k:][::-1])

    layer_top_k = np.zeros((n_layers + 1, top_k), dtype=int)
    agreement = np.zeros(n_layers)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            layer_logits = jnp.dot(resid[pos], W_U)
            if b_U is not None:
                layer_logits = layer_logits + b_U
            top = np.array(jnp.argsort(layer_logits)[-top_k:][::-1])
            layer_top_k[layer] = top
            if layer > 0:
                overlap = len(set(top.tolist()) & set(final_top_k.tolist()))
                agreement[layer - 1] = overlap / top_k

    # First agreement on top-1
    final_top1 = int(final_top_k[0])
    first_agree = n_layers
    for l in range(n_layers + 1):
        if layer_top_k[l, 0] == final_top1:
            first_agree = l
            break

    consensus = np.mean([1 if layer_top_k[l, 0] == final_top1 else 0 for l in range(n_layers + 1)])

    return {
        "layer_top_k": layer_top_k,
        "agreement_with_final": agreement,
        "first_agreement_layer": first_agree,
        "consensus_fraction": float(consensus),
    }


def rank_trajectory(model, tokens, target_tokens, pos=-1):
    """Track the rank of specific tokens through the layers.

    Shows how certain tokens rise or fall in the ranking as computation proceeds.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_tokens: List of token indices to track.
        pos: Position.

    Returns:
        dict with:
            ranks: dict mapping target_token -> array [n_layers+1] of ranks
            logits: dict mapping target_token -> array [n_layers+1] of logit values
            final_ranks: dict mapping target_token -> final rank
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    W_U = model.unembed.W_U
    b_U = getattr(model.unembed, 'b_U', None)

    ranks = {}
    logit_vals = {}

    for t in target_tokens:
        t = int(t)
        ranks[t] = np.zeros(n_layers + 1, dtype=int)
        logit_vals[t] = np.zeros(n_layers + 1)

        for layer in range(n_layers + 1):
            if layer == 0:
                key = "blocks.0.hook_resid_pre"
            else:
                key = f"blocks.{layer - 1}.hook_resid_post"
            resid = cache.get(key)
            if resid is not None:
                layer_logits = jnp.dot(resid[pos], W_U)
                if b_U is not None:
                    layer_logits = layer_logits + b_U
                logit_vals[t][layer] = float(layer_logits[t])
                ranks[t][layer] = int(jnp.sum(layer_logits > layer_logits[t]))

    final_ranks = {int(t): int(ranks[int(t)][-1]) for t in target_tokens}

    return {
        "ranks": ranks,
        "logits": logit_vals,
        "final_ranks": final_ranks,
    }
