"""Token-level ablation and causal effects.

Per-token knockout, necessity/sufficiency testing, minimal token set
identification, pairwise token interaction measurement, and token
importance ranking.
"""

import jax
import jax.numpy as jnp
import numpy as np


def per_token_knockout(model, tokens, metric_fn, replacement_token=0):
    """Measure the effect of replacing each input token.

    For each position, replace the token with a default token and
    measure the change in the metric.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        replacement_token: Token to substitute (default 0).

    Returns:
        dict with:
            token_effects: [seq_len] metric change when each token is replaced
            most_important_positions: list of (position, effect)
            least_important_positions: list of (position, effect)
            base_metric: float
    """
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    effects = np.zeros(seq_len)

    for pos in range(seq_len):
        modified = tokens_np.copy()
        modified[pos] = replacement_token
        mod_logits = np.array(model(jnp.array(modified)))
        effects[pos] = base_metric - float(metric_fn(mod_logits))

    # Sort by importance
    sorted_idx = np.argsort(-np.abs(effects))
    most_important = [(int(i), float(effects[i])) for i in sorted_idx[:10]]
    least_important = [(int(i), float(effects[i])) for i in sorted_idx[-10:]]

    return {
        "token_effects": effects,
        "most_important_positions": most_important,
        "least_important_positions": least_important,
        "base_metric": base_metric,
    }


def token_necessity_sufficiency(model, tokens, metric_fn, threshold=0.5, replacement_token=0):
    """Test necessity and sufficiency of each token position.

    Necessity: how much does removing the token hurt the metric?
    Sufficiency: how much of the metric is recovered with only this token?

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        threshold: Fraction of base metric to consider sufficient.
        replacement_token: Token for replacement.

    Returns:
        dict with:
            necessity: [seq_len] necessity score per position
            sufficiency: [seq_len] sufficiency score per position
            necessary_positions: list of int (positions whose removal drops metric below threshold)
            sufficient_positions: list of int (positions that alone achieve threshold)
            necessity_sufficiency_correlation: float
    """
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # All-replaced baseline
    all_replaced = np.full(seq_len, replacement_token, dtype=tokens_np.dtype)
    null_logits = np.array(model(jnp.array(all_replaced)))
    null_metric = float(metric_fn(null_logits))

    necessity = np.zeros(seq_len)
    sufficiency = np.zeros(seq_len)

    for pos in range(seq_len):
        # Necessity: remove this token
        removed = tokens_np.copy()
        removed[pos] = replacement_token
        rem_logits = np.array(model(jnp.array(removed)))
        rem_metric = float(metric_fn(rem_logits))
        # How much does removing hurt? (normalized)
        if abs(base_metric - null_metric) > 1e-10:
            necessity[pos] = (base_metric - rem_metric) / abs(base_metric - null_metric)
        else:
            necessity[pos] = 0.0

        # Sufficiency: only this token present
        only_this = all_replaced.copy()
        only_this[pos] = tokens_np[pos]
        only_logits = np.array(model(jnp.array(only_this)))
        only_metric = float(metric_fn(only_logits))
        if abs(base_metric - null_metric) > 1e-10:
            sufficiency[pos] = (only_metric - null_metric) / abs(base_metric - null_metric)
        else:
            sufficiency[pos] = 0.0

    necessary_pos = [int(i) for i in range(seq_len) if necessity[i] > threshold]
    sufficient_pos = [int(i) for i in range(seq_len) if sufficiency[i] > threshold]

    # Correlation
    if np.std(necessity) > 1e-10 and np.std(sufficiency) > 1e-10:
        corr = float(np.corrcoef(necessity, sufficiency)[0, 1])
    else:
        corr = 0.0

    return {
        "necessity": necessity,
        "sufficiency": sufficiency,
        "necessary_positions": necessary_pos,
        "sufficient_positions": sufficient_pos,
        "necessity_sufficiency_correlation": corr,
    }


def minimal_token_set(model, tokens, metric_fn, threshold=0.8, replacement_token=0):
    """Find the minimal set of tokens needed to preserve the metric.

    Uses a greedy approach: start with all tokens replaced, then add
    back tokens one at a time in order of importance until the metric
    exceeds the threshold.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        threshold: Fraction of base metric to achieve.
        replacement_token: Token for replacement.

    Returns:
        dict with:
            minimal_set: list of int (positions in the minimal set)
            set_size: int
            metric_achieved: float
            metric_trajectory: list of (position_added, metric_value)
            coverage: float (fraction of base metric achieved)
    """
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # Start with all replaced
    current = np.full(seq_len, replacement_token, dtype=tokens_np.dtype)
    current_logits = np.array(model(jnp.array(current)))
    current_metric = float(metric_fn(current_logits))

    target = base_metric * threshold if base_metric > 0 else base_metric + abs(base_metric) * (1 - threshold)

    minimal_set = []
    trajectory = []
    remaining = list(range(seq_len))

    while remaining and (base_metric > 0 and current_metric < target or
                          base_metric <= 0 and current_metric > target):
        # Try each remaining position
        best_pos = None
        best_metric = current_metric
        best_improvement = -float('inf')

        for pos in remaining:
            trial = current.copy()
            trial[pos] = tokens_np[pos]
            trial_logits = np.array(model(jnp.array(trial)))
            trial_metric = float(metric_fn(trial_logits))

            improvement = trial_metric - current_metric
            if base_metric < 0:
                improvement = -improvement

            if improvement > best_improvement:
                best_improvement = improvement
                best_pos = pos
                best_metric = trial_metric

        if best_pos is None:
            break

        current[best_pos] = tokens_np[best_pos]
        current_metric = best_metric
        minimal_set.append(best_pos)
        remaining.remove(best_pos)
        trajectory.append((best_pos, current_metric))

    coverage = current_metric / (base_metric + 1e-10) if base_metric != 0 else 0.0

    return {
        "minimal_set": minimal_set,
        "set_size": len(minimal_set),
        "metric_achieved": current_metric,
        "metric_trajectory": trajectory,
        "coverage": float(np.clip(coverage, 0, 2)),
    }


def pairwise_token_interaction(model, tokens, metric_fn, replacement_token=0):
    """Measure pairwise interaction effects between tokens.

    For each pair of positions, measure whether their joint effect
    differs from the sum of individual effects (synergy/redundancy).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        replacement_token: Token for replacement.

    Returns:
        dict with:
            interaction_matrix: [seq_len, seq_len] pairwise interaction strengths
            synergistic_pairs: list of (pos_i, pos_j, interaction)
            redundant_pairs: list of (pos_i, pos_j, interaction)
            max_interaction: (pos_i, pos_j, value)
    """
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # Single ablation effects
    single_effects = np.zeros(seq_len)
    for pos in range(seq_len):
        modified = tokens_np.copy()
        modified[pos] = replacement_token
        mod_logits = np.array(model(jnp.array(modified)))
        single_effects[pos] = base_metric - float(metric_fn(mod_logits))

    # Pairwise ablation effects
    interaction = np.zeros((seq_len, seq_len))
    synergistic = []
    redundant = []

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            modified = tokens_np.copy()
            modified[i] = replacement_token
            modified[j] = replacement_token
            pair_logits = np.array(model(jnp.array(modified)))
            pair_effect = base_metric - float(metric_fn(pair_logits))

            # Interaction = joint - sum of individual
            expected = single_effects[i] + single_effects[j]
            inter = pair_effect - expected
            interaction[i, j] = inter
            interaction[j, i] = inter

            if inter > 0.01:
                synergistic.append((i, j, float(inter)))
            elif inter < -0.01:
                redundant.append((i, j, float(inter)))

    synergistic.sort(key=lambda x: -x[2])
    redundant.sort(key=lambda x: x[2])

    # Max interaction
    flat_idx = np.unravel_index(np.argmax(np.abs(interaction)), interaction.shape)
    max_inter = (int(flat_idx[0]), int(flat_idx[1]),
                 float(interaction[flat_idx[0], flat_idx[1]]))

    return {
        "interaction_matrix": interaction,
        "synergistic_pairs": synergistic[:10],
        "redundant_pairs": redundant[:10],
        "max_interaction": max_inter,
    }


def token_importance_ranking(model, tokens, metric_fn, method="knockout", replacement_token=0):
    """Rank tokens by importance using multiple methods.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.
        method: "knockout" (replace with default) or "leave_one_in" (only keep one).
        replacement_token: Token for replacement.

    Returns:
        dict with:
            ranking: list of (position, importance_score) sorted by importance
            importance_scores: [seq_len] raw importance scores
            normalized_scores: [seq_len] normalized to sum to 1
            entropy: float (how spread out the importance is)
    """
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    scores = np.zeros(seq_len)

    if method == "knockout":
        for pos in range(seq_len):
            modified = tokens_np.copy()
            modified[pos] = replacement_token
            mod_logits = np.array(model(jnp.array(modified)))
            scores[pos] = abs(base_metric - float(metric_fn(mod_logits)))
    elif method == "leave_one_in":
        all_replaced = np.full(seq_len, replacement_token, dtype=tokens_np.dtype)
        null_logits = np.array(model(jnp.array(all_replaced)))
        null_metric = float(metric_fn(null_logits))
        for pos in range(seq_len):
            only_this = all_replaced.copy()
            only_this[pos] = tokens_np[pos]
            only_logits = np.array(model(jnp.array(only_this)))
            scores[pos] = abs(float(metric_fn(only_logits)) - null_metric)

    # Ranking
    sorted_idx = np.argsort(-scores)
    ranking = [(int(i), float(scores[i])) for i in sorted_idx]

    # Normalize
    total = np.sum(scores)
    if total > 1e-10:
        normalized = scores / total
    else:
        normalized = np.ones(seq_len) / seq_len

    # Entropy
    p = normalized[normalized > 1e-10]
    entropy = -float(np.sum(p * np.log(p)))

    return {
        "ranking": ranking,
        "importance_scores": scores,
        "normalized_scores": normalized,
        "entropy": entropy,
    }
