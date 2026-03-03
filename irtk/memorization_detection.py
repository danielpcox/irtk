"""Memorization detection: analyzing memorization vs generalization.

Detects signatures of memorization in model internals — distinguishing
memorized facts from genuinely learned patterns. Identifies which layers
store memorized content and how extractable memorized data is.

Functions:
- memorization_score: Score how likely a sequence is memorized vs generalized
- extractability_by_layer: Measure extractability of target info at each layer
- generalization_gap_profile: Per-layer generalization gap analysis
- memorized_token_localization: Locate which positions trigger memorized recall
- content_extraction_risk: Assess risk of memorized content being extractable

References:
    - Carlini et al. (2023) "Extracting Training Data from Large Language Models"
    - Tirumala et al. (2022) "Memorization Without Overfitting"
    - Biderman et al. (2023) "Pythia: Suite for Analyzing LLMs Across Training"
"""

from typing import Optional, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def memorization_score(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    n_perturbations: int = 5,
    seed: int = 42,
) -> dict:
    """Score how likely a sequence is memorized vs generalized.

    Memorized sequences show high confidence that drops sharply with
    small perturbations, while generalized patterns degrade gracefully.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        n_perturbations: Number of perturbed versions to test.
        seed: Random seed.

    Returns:
        Dict with:
            "clean_confidence": confidence on original sequence
            "perturbed_confidences": [n_perturbations] confidence on each perturbation
            "confidence_drop": mean confidence decrease
            "memorization_score": ratio of drop to clean confidence (higher = more memorized)
            "is_likely_memorized": whether score exceeds typical threshold
    """
    logits = np.array(model(tokens))
    clean_probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    clean_probs = clean_probs / np.sum(clean_probs, axis=-1, keepdims=True)

    # Clean confidence: mean max probability
    clean_conf = float(np.mean(np.max(clean_probs, axis=-1)))

    rng = np.random.RandomState(seed)
    perturbed_confs = []

    for _ in range(n_perturbations):
        # Perturb: replace one random token
        perturbed = np.array(tokens)
        pos = rng.randint(0, len(tokens))
        perturbed[pos] = rng.randint(0, model.cfg.d_vocab)
        perturbed = jnp.array(perturbed)

        p_logits = np.array(model(perturbed))
        p_probs = np.exp(p_logits - np.max(p_logits, axis=-1, keepdims=True))
        p_probs = p_probs / np.sum(p_probs, axis=-1, keepdims=True)
        perturbed_confs.append(float(np.mean(np.max(p_probs, axis=-1))))

    perturbed_confs = np.array(perturbed_confs)
    mean_drop = clean_conf - float(np.mean(perturbed_confs))

    # Score: normalized drop (high = memorized, drops sharply)
    score = mean_drop / (clean_conf + 1e-10)
    score = max(0.0, score)

    return {
        "clean_confidence": clean_conf,
        "perturbed_confidences": perturbed_confs,
        "confidence_drop": mean_drop,
        "memorization_score": score,
        "is_likely_memorized": score > 0.5,
    }


def extractability_by_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict:
    """Measure extractability of target info at each layer.

    Uses logit lens to measure how much each layer "knows" about the
    target token, revealing where memorized content becomes accessible.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token to measure extractability for.
        pos: Position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_ranks": [n_layers] rank of target token in logit lens prediction
            "layer_probs": [n_layers] probability of target at each layer
            "extraction_layer": first layer where target enters top-10
            "final_rank": rank in final output
            "accessibility_curve": [n_layers] normalized extractability (0=inaccessible, 1=top)
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    ranks = np.zeros(n_layers, dtype=int)
    probs = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            layer_probs = np.exp(layer_logits - np.max(layer_logits))
            layer_probs = layer_probs / np.sum(layer_probs)

            probs[l] = float(layer_probs[target_token])
            # Rank (0 = top)
            ranks[l] = int(np.sum(layer_logits > layer_logits[target_token]))

    # Extraction layer: first where rank < 10
    extraction = -1
    for l in range(n_layers):
        if ranks[l] < 10:
            extraction = l
            break

    # Final rank
    final_logits = np.array(logits[pos])
    final_rank = int(np.sum(final_logits > final_logits[target_token]))

    # Accessibility: 1 - normalized rank
    d_vocab = model.cfg.d_vocab
    accessibility = 1.0 - ranks / (d_vocab + 1e-10)

    return {
        "layer_ranks": ranks,
        "layer_probs": probs,
        "extraction_layer": extraction,
        "final_rank": final_rank,
        "accessibility_curve": accessibility,
    }


def generalization_gap_profile(
    model: HookedTransformer,
    train_tokens: jnp.ndarray,
    test_tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Per-layer generalization gap analysis.

    Compares the model's internal predictions on training-like vs test-like
    inputs at each layer to identify where memorization diverges from generalization.

    Args:
        model: HookedTransformer.
        train_tokens: [seq_len] training-like tokens.
        test_tokens: [seq_len] test-like tokens.
        pos: Position to analyze (-1 = last).

    Returns:
        Dict with:
            "train_entropies": [n_layers] prediction entropy on training tokens
            "test_entropies": [n_layers] prediction entropy on test tokens
            "entropy_gap": [n_layers] difference in entropies (train - test)
            "max_gap_layer": layer with largest generalization gap
            "representation_distance": [n_layers] cosine distance between train/test representations
    """
    _, cache_train = model.run_with_cache(train_tokens)
    _, cache_test = model.run_with_cache(test_tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    train_ent = np.zeros(n_layers)
    test_ent = np.zeros(n_layers)
    rep_dist = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"

        for tokens_cache, ent_arr in [(cache_train, train_ent), (cache_test, test_ent)]:
            if key in tokens_cache.cache_dict:
                resid = np.array(tokens_cache.cache_dict[key][pos])
                logits = resid @ W_U + b_U
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                ent_arr[l] = -float(np.sum(probs * np.log(probs + 1e-10)))

        # Representation distance
        if key in cache_train.cache_dict and key in cache_test.cache_dict:
            r_train = np.array(cache_train.cache_dict[key][pos])
            r_test = np.array(cache_test.cache_dict[key][pos])
            cos = np.dot(r_train, r_test) / (np.linalg.norm(r_train) * np.linalg.norm(r_test) + 1e-10)
            rep_dist[l] = 1.0 - cos

    gap = train_ent - test_ent
    max_gap_layer = int(np.argmax(np.abs(gap)))

    return {
        "train_entropies": train_ent,
        "test_entropies": test_ent,
        "entropy_gap": gap,
        "max_gap_layer": max_gap_layer,
        "representation_distance": rep_dist,
    }


def memorized_token_localization(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Locate which token positions trigger memorized recall.

    Replaces each position with a random token and measures the effect
    on the output, identifying trigger positions for memorized content.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "position_importance": [seq_len] metric change when each position is replaced
            "trigger_positions": positions with importance > 2x mean
            "most_critical_position": position causing largest metric change
            "importance_concentration": fraction of total importance in trigger positions
    """
    clean_logits = model(tokens)
    clean_metric = float(metric_fn(clean_logits))

    seq_len = len(tokens)
    importance = np.zeros(seq_len)

    for pos in range(seq_len):
        # Replace with token 0 (arbitrary replacement)
        modified = np.array(tokens)
        modified[pos] = 0 if tokens[pos] != 0 else 1
        modified_logits = model(jnp.array(modified))
        importance[pos] = abs(float(metric_fn(modified_logits)) - clean_metric)

    mean_imp = float(np.mean(importance))
    triggers = [int(p) for p in range(seq_len) if importance[p] > 2.0 * mean_imp]
    most_critical = int(np.argmax(importance))

    total = float(np.sum(importance))
    trigger_sum = float(np.sum(importance[triggers])) if triggers else 0.0
    concentration = trigger_sum / (total + 1e-10)

    return {
        "position_importance": importance,
        "trigger_positions": triggers,
        "most_critical_position": most_critical,
        "importance_concentration": concentration,
    }


def content_extraction_risk(
    model: HookedTransformer,
    prefix_tokens: jnp.ndarray,
    target_tokens: jnp.ndarray,
) -> dict:
    """Assess risk of memorized content being extractable.

    Given a prefix and expected continuation, measures how confidently
    the model predicts the target tokens, indicating memorization risk.

    Args:
        model: HookedTransformer.
        prefix_tokens: [prefix_len] prefix token sequence.
        target_tokens: [target_len] expected continuation tokens.

    Returns:
        Dict with:
            "per_token_probs": [target_len] probability of each target token
            "mean_probability": mean probability across target tokens
            "exact_match_prob": probability of generating exact sequence (product)
            "extraction_risk": qualitative risk level based on mean probability
            "weakest_link": position with lowest target token probability
    """
    # Build full sequence
    full = jnp.concatenate([prefix_tokens, target_tokens])
    logits = np.array(model(full))

    prefix_len = len(prefix_tokens)
    target_len = len(target_tokens)

    per_token_probs = np.zeros(target_len)

    for i in range(target_len):
        # Probability of target[i] given prefix + target[:i]
        pos = prefix_len + i - 1  # Position that should predict target[i]
        if pos >= 0 and pos < len(logits):
            token_logits = logits[pos]
            probs = np.exp(token_logits - np.max(token_logits))
            probs = probs / np.sum(probs)
            per_token_probs[i] = float(probs[target_tokens[i]])

    mean_prob = float(np.mean(per_token_probs))

    # Exact match probability (product, in log space for stability)
    log_probs = np.log(per_token_probs + 1e-30)
    exact_prob = float(np.exp(np.sum(log_probs)))

    # Risk assessment
    if mean_prob > 0.8:
        risk = "high"
    elif mean_prob > 0.3:
        risk = "medium"
    else:
        risk = "low"

    weakest = int(np.argmin(per_token_probs))

    return {
        "per_token_probs": per_token_probs,
        "mean_probability": mean_prob,
        "exact_match_prob": exact_prob,
        "extraction_risk": risk,
        "weakest_link": weakest,
    }
