"""Output logit analysis: detailed analysis of the final logit distribution."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def logit_distribution_profile(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Profile the output logit distribution at a given position.

    Analyzes distribution shape: mean, std, skewness, top/bottom tokens.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    l = logits[pos]  # [d_vocab]
    mean = float(jnp.mean(l))
    std = float(jnp.std(l))
    skewness = float(jnp.mean(((l - mean) / (std + 1e-10)) ** 3))

    probs = jax.nn.softmax(l)
    entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-10)))

    top_5 = jnp.argsort(l)[-5:][::-1]
    bottom_5 = jnp.argsort(l)[:5]

    return {
        'position': pos,
        'mean_logit': mean,
        'std_logit': std,
        'skewness': skewness,
        'entropy': entropy,
        'top_tokens': [{'token': int(t), 'logit': float(l[t]), 'probability': float(probs[t])} for t in top_5],
        'bottom_tokens': [{'token': int(t), 'logit': float(l[t]), 'probability': float(probs[t])} for t in bottom_5],
    }


def logit_temperature_sensitivity(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How sensitive is the prediction to temperature scaling?

    Tests different temperatures and measures prediction stability.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    l = logits[pos]
    base_probs = jax.nn.softmax(l)
    base_top = int(jnp.argmax(base_probs))

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    per_temperature = []

    for temp in temperatures:
        scaled_probs = jax.nn.softmax(l / temp)
        top = int(jnp.argmax(scaled_probs))
        conf = float(scaled_probs[top])
        entropy = float(-jnp.sum(scaled_probs * jnp.log(scaled_probs + 1e-10)))

        per_temperature.append({
            'temperature': temp,
            'top_token': top,
            'confidence': conf,
            'entropy': entropy,
            'same_prediction': top == base_top,
        })

    n_stable = sum(1 for p in per_temperature if p['same_prediction'])

    return {
        'position': pos,
        'base_prediction': base_top,
        'per_temperature': per_temperature,
        'n_stable_temperatures': n_stable,
        'is_robust': n_stable == len(temperatures),
    }


def logit_margin_analysis(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze the margin between top-1 and top-2 predictions at each position.

    Large margin = confident; small margin = uncertain.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]

    per_position = []
    for pos in range(seq_len):
        l = logits[pos]
        sorted_logits = jnp.sort(l)[::-1]
        top1 = int(jnp.argmax(l))
        top1_logit = float(sorted_logits[0])
        top2_logit = float(sorted_logits[1])
        margin = top1_logit - top2_logit

        probs = jax.nn.softmax(l)
        prob_margin = float(probs[top1]) - float(jnp.sort(probs)[-2])

        per_position.append({
            'position': pos,
            'top1_token': top1,
            'top1_logit': top1_logit,
            'top2_logit': top2_logit,
            'logit_margin': margin,
            'probability_margin': prob_margin,
            'is_decisive': margin > 1.0,
        })

    mean_margin = sum(p['logit_margin'] for p in per_position) / len(per_position)
    n_decisive = sum(1 for p in per_position if p['is_decisive'])

    return {
        'per_position': per_position,
        'mean_logit_margin': mean_margin,
        'n_decisive': n_decisive,
    }


def logit_rank_distribution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Analyze the rank distribution of logits.

    How concentrated are the logits? Do a few tokens dominate?
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    l = logits[pos]
    probs = jax.nn.softmax(l)
    sorted_probs = jnp.sort(probs)[::-1]

    # Cumulative probability
    cumulative = jnp.cumsum(sorted_probs)
    top_1_prob = float(sorted_probs[0])
    top_5_prob = float(cumulative[min(4, len(cumulative)-1)])
    top_10_prob = float(cumulative[min(9, len(cumulative)-1)])

    # Tokens needed for 50%, 90%, 95%
    tokens_50 = int(jnp.searchsorted(cumulative, 0.5) + 1)
    tokens_90 = int(jnp.searchsorted(cumulative, 0.9) + 1)
    tokens_95 = int(jnp.searchsorted(cumulative, 0.95) + 1)

    return {
        'position': pos,
        'top_1_probability': top_1_prob,
        'top_5_probability': top_5_prob,
        'top_10_probability': top_10_prob,
        'tokens_for_50_pct': tokens_50,
        'tokens_for_90_pct': tokens_90,
        'tokens_for_95_pct': tokens_95,
        'is_concentrated': tokens_50 <= 3,
    }


def cross_position_logit_consistency(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How consistent are logit distributions across positions?

    Measures pairwise KL divergence between positions.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]

    probs = jax.nn.softmax(logits, axis=-1)  # [seq, d_vocab]
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    per_position = []
    total_kl = 0.0
    n_pairs = 0

    for i in range(seq_len):
        kls = []
        for j in range(seq_len):
            if i != j:
                kl = float(jnp.sum(probs[i] * (log_probs[i] - log_probs[j])))
                kls.append(kl)
                total_kl += kl
                n_pairs += 1

        mean_kl = sum(kls) / len(kls) if kls else 0.0
        per_position.append({
            'position': i,
            'mean_kl_to_others': mean_kl,
            'is_outlier': mean_kl > 1.0,
        })

    mean_kl = total_kl / n_pairs if n_pairs > 0 else 0.0

    return {
        'per_position': per_position,
        'mean_pairwise_kl': mean_kl,
        'is_consistent': mean_kl < 1.0,
        'n_outliers': sum(1 for p in per_position if p['is_outlier']),
    }
