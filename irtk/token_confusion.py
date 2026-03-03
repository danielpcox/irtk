"""Token confusion analysis: systematic prediction error patterns.

Tools for analyzing which tokens the model confuses with each other:
- Confusion matrix at specific positions/layers
- Systematic error patterns (which tokens are consistently mixed up)
- Logit competition (when two tokens are close in prediction)
- Layer-resolved confusion tracking
- Position-dependent error modes
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def prediction_confusion_matrix(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Build a confusion matrix of actual vs predicted tokens.

    For each sequence, looks at the target token at `pos+1` and what the
    model actually predicts at `pos`. Aggregates into a confusion matrix.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences (each at least |pos|+2 long).
        pos: Position to analyze predictions at.
        top_k: Number of top confused pairs to return.

    Returns:
        Dict with confusion matrix, top confused pairs, accuracy.
    """
    confusion = {}  # (actual, predicted) -> count
    correct = 0
    total = 0

    for tokens in tokens_list:
        seq_len = len(tokens)
        actual_pos = pos if pos >= 0 else seq_len + pos
        if actual_pos + 1 >= seq_len:
            continue

        logits = model(tokens)
        predicted = int(jnp.argmax(logits[actual_pos]))
        actual = int(tokens[actual_pos + 1])

        pair = (actual, predicted)
        confusion[pair] = confusion.get(pair, 0) + 1
        if predicted == actual:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Find top confused pairs (excluding correct predictions)
    wrong_pairs = [(k, v) for k, v in confusion.items() if k[0] != k[1]]
    wrong_pairs.sort(key=lambda x: -x[1])

    top_confused = []
    for (actual, predicted), count in wrong_pairs[:top_k]:
        top_confused.append({
            'actual': actual,
            'predicted': predicted,
            'count': count,
        })

    return {
        'accuracy': round(accuracy, 4),
        'total_predictions': total,
        'n_confused_pairs': len(wrong_pairs),
        'top_confused': top_confused,
    }


def logit_competition(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Analyze competition between top token predictions.

    Measures how close the top predictions are to each other, indicating
    uncertainty or competition between candidates.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of top tokens to compare.

    Returns:
        Dict with top predictions, gaps, and competition score.
    """
    logits = model(tokens)
    logits_at_pos = np.array(logits[pos])  # [d_vocab]

    top_indices = np.argsort(logits_at_pos)[::-1][:top_k]
    top_logits = logits_at_pos[top_indices]

    # Competition score: ratio of 2nd best to 1st best probability
    probs = np.exp(logits_at_pos - logits_at_pos.max())
    probs = probs / probs.sum()
    top_probs = probs[top_indices]

    if top_probs[0] > 1e-10:
        competition_score = float(top_probs[1] / top_probs[0]) if len(top_probs) > 1 else 0.0
    else:
        competition_score = 0.0

    top_tokens = []
    for i, idx in enumerate(top_indices):
        gap_from_top = float(top_logits[0] - top_logits[i])
        top_tokens.append({
            'token': int(idx),
            'logit': round(float(top_logits[i]), 4),
            'probability': round(float(top_probs[i]), 4),
            'gap_from_top': round(gap_from_top, 4),
        })

    return {
        'top_tokens': top_tokens,
        'competition_score': round(competition_score, 4),
        'entropy': round(-float(np.sum(probs * np.log(probs + 1e-10))), 4),
    }


def layer_resolved_confusion(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Track how the top predictions change across layers.

    Applies the logit lens at each layer to see when predictions
    converge and when they switch.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of top predictions per layer.

    Returns:
        Dict with per-layer top predictions and transition points.
    """
    _, cache = model.run_with_cache(tokens)
    resid_stack = cache.accumulated_resid()  # [n_components, seq, d_model]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_top = None
    transitions = []

    for i in range(resid_stack.shape[0]):
        resid = resid_stack[i, pos]

        # Apply layer norm if available
        if model.ln_final is not None:
            # Expand to [1, d_model] for layer norm
            resid_expanded = resid[None, :]
            normed = model.ln_final(resid_expanded)[0]
        else:
            normed = resid

        logits = jnp.dot(normed, W_U) + b_U  # [d_vocab]
        logits_np = np.array(logits)

        top_indices = np.argsort(logits_np)[::-1][:top_k]
        top_vals = logits_np[top_indices]

        current_top = int(top_indices[0])
        if prev_top is not None and current_top != prev_top:
            transitions.append({
                'stage': i,
                'from_token': prev_top,
                'to_token': current_top,
            })
        prev_top = current_top

        per_layer.append({
            'stage': i,
            'top_token': current_top,
            'top_logit': round(float(top_vals[0]), 4),
            'runner_up': int(top_indices[1]) if len(top_indices) > 1 else -1,
            'gap': round(float(top_vals[0] - top_vals[1]), 4) if len(top_vals) > 1 else 0.0,
        })

    return {
        'per_layer': per_layer,
        'n_transitions': len(transitions),
        'transitions': transitions,
        'final_prediction': per_layer[-1]['top_token'] if per_layer else -1,
    }


def systematic_errors(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
    pos: int = -1,
    min_occurrences: int = 2,
) -> dict:
    """Find tokens that are systematically predicted wrong.

    Identifies tokens that the model consistently confuses across
    different contexts.

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences.
        pos: Position to analyze.
        min_occurrences: Minimum times a token must appear to count.

    Returns:
        Dict with per-token error rates and systematic error patterns.
    """
    token_stats = {}  # token_id -> {'correct': int, 'total': int, 'predicted_as': dict}

    for tokens in tokens_list:
        seq_len = len(tokens)
        actual_pos = pos if pos >= 0 else seq_len + pos
        if actual_pos + 1 >= seq_len:
            continue

        logits = model(tokens)
        predicted = int(jnp.argmax(logits[actual_pos]))
        actual = int(tokens[actual_pos + 1])

        if actual not in token_stats:
            token_stats[actual] = {'correct': 0, 'total': 0, 'predicted_as': {}}

        token_stats[actual]['total'] += 1
        if predicted == actual:
            token_stats[actual]['correct'] += 1
        else:
            token_stats[actual]['predicted_as'][predicted] = \
                token_stats[actual]['predicted_as'].get(predicted, 0) + 1

    # Filter by min occurrences and sort by error rate
    results = []
    for token_id, stats in token_stats.items():
        if stats['total'] >= min_occurrences:
            error_rate = 1.0 - stats['correct'] / stats['total']
            # Top confused-with token
            if stats['predicted_as']:
                top_confused = max(stats['predicted_as'], key=stats['predicted_as'].get)
                top_confused_count = stats['predicted_as'][top_confused]
            else:
                top_confused = -1
                top_confused_count = 0

            results.append({
                'token': token_id,
                'total': stats['total'],
                'error_rate': round(error_rate, 4),
                'top_confused_with': top_confused,
                'top_confused_count': top_confused_count,
            })

    results.sort(key=lambda x: -x['error_rate'])

    return {
        'per_token': results[:20],
        'n_tokens_analyzed': len(results),
        'mean_error_rate': round(float(np.mean([r['error_rate'] for r in results])), 4) if results else 0.0,
    }


def position_error_modes(
    model: HookedTransformer,
    tokens_list: list[jnp.ndarray],
) -> dict:
    """Analyze how prediction accuracy varies by position in the sequence.

    Some positions are inherently harder to predict (e.g., the first few tokens
    or tokens at sentence boundaries).

    Args:
        model: HookedTransformer.
        tokens_list: List of token sequences (should be same length).

    Returns:
        Dict with per-position accuracy and error characteristics.
    """
    if not tokens_list:
        return {'per_position': [], 'n_examples': 0}

    seq_len = len(tokens_list[0])
    position_correct = np.zeros(seq_len - 1)
    position_total = np.zeros(seq_len - 1)
    position_entropy = np.zeros(seq_len - 1)

    for tokens in tokens_list:
        if len(tokens) != seq_len:
            continue

        logits = model(tokens)

        for pos in range(seq_len - 1):
            predicted = int(jnp.argmax(logits[pos]))
            actual = int(tokens[pos + 1])

            position_total[pos] += 1
            if predicted == actual:
                position_correct[pos] += 1

            # Entropy at this position
            probs = jax.nn.softmax(logits[pos])
            ent = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
            position_entropy[pos] += ent

    per_position = []
    for pos in range(seq_len - 1):
        if position_total[pos] > 0:
            acc = position_correct[pos] / position_total[pos]
            avg_ent = position_entropy[pos] / position_total[pos]
        else:
            acc = 0.0
            avg_ent = 0.0

        per_position.append({
            'position': pos,
            'accuracy': round(float(acc), 4),
            'avg_entropy': round(float(avg_ent), 4),
            'n_examples': int(position_total[pos]),
        })

    accuracies = [p['accuracy'] for p in per_position if p['n_examples'] > 0]
    return {
        'per_position': per_position,
        'n_examples': len(tokens_list),
        'mean_accuracy': round(float(np.mean(accuracies)), 4) if accuracies else 0.0,
        'best_position': int(np.argmax(accuracies)) if accuracies else -1,
        'worst_position': int(np.argmin(accuracies)) if accuracies else -1,
    }
