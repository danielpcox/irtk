"""Prediction landscape analysis: logit distribution properties and decision margins."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def logit_distribution_profile(model: HookedTransformer, tokens: jnp.ndarray,
                                 position: int = -1) -> dict:
    """Statistical profile of the output logit distribution.

    Shows how peaked, spread, and skewed the predictions are.
    """
    logits = model(tokens)  # [seq, d_vocab]
    if position < 0:
        position = logits.shape[0] + position
    logit_vec = logits[position]  # [d_vocab]

    mean_logit = float(jnp.mean(logit_vec))
    std_logit = float(jnp.std(logit_vec))
    max_logit = float(jnp.max(logit_vec))
    min_logit = float(jnp.min(logit_vec))

    probs = jnp.exp(logit_vec - jnp.max(logit_vec))
    probs = probs / jnp.sum(probs)
    entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))

    top_idx = int(jnp.argmax(logit_vec))

    return {
        "position": position,
        "mean": mean_logit,
        "std": std_logit,
        "max": max_logit,
        "min": min_logit,
        "range": max_logit - min_logit,
        "entropy": entropy,
        "top_token": top_idx,
        "top_logit": max_logit,
    }


def decision_margin_analysis(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1, top_k: int = 5) -> dict:
    """Margin between the top prediction and alternatives.

    Large margin = confident decision; small margin = ambiguous.
    """
    logits = model(tokens)
    if position < 0:
        position = logits.shape[0] + position
    logit_vec = logits[position]

    sorted_indices = jnp.argsort(-logit_vec)
    top_tokens = []
    top_logits = []
    for i in range(min(top_k, logit_vec.shape[0])):
        idx = int(sorted_indices[i])
        top_tokens.append(idx)
        top_logits.append(float(logit_vec[idx]))

    margin_1_2 = top_logits[0] - top_logits[1] if len(top_logits) > 1 else float('inf')
    margin_1_3 = top_logits[0] - top_logits[2] if len(top_logits) > 2 else float('inf')

    return {
        "position": position,
        "top_tokens": top_tokens,
        "top_logits": top_logits,
        "margin_1_2": margin_1_2,
        "margin_1_3": margin_1_3,
        "is_confident": margin_1_2 > 2.0,
    }


def prediction_entropy_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Entropy of predictions at each position.

    Shows which positions are easy (low entropy) vs hard (high entropy).
    """
    logits = model(tokens)  # [seq, d_vocab]
    seq_len = logits.shape[0]

    per_position = []
    for pos in range(seq_len):
        lv = logits[pos]
        probs = jnp.exp(lv - jnp.max(lv))
        probs = probs / jnp.sum(probs)
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))
        top_prob = float(jnp.max(probs))
        per_position.append({
            "position": pos,
            "entropy": entropy,
            "top_prob": top_prob,
        })
    entropies = [p["entropy"] for p in per_position]
    return {
        "per_position": per_position,
        "mean_entropy": sum(entropies) / len(entropies),
        "min_entropy_pos": min(range(len(entropies)), key=lambda i: entropies[i]),
        "max_entropy_pos": max(range(len(entropies)), key=lambda i: entropies[i]),
    }


def logit_concentration(model: HookedTransformer, tokens: jnp.ndarray,
                          position: int = -1) -> dict:
    """How concentrated the probability mass is among top tokens.

    Measures top-1, top-5, top-10 cumulative probability.
    """
    logits = model(tokens)
    if position < 0:
        position = logits.shape[0] + position
    logit_vec = logits[position]

    probs = jnp.exp(logit_vec - jnp.max(logit_vec))
    probs = probs / jnp.sum(probs)
    sorted_probs = jnp.sort(probs)[::-1]

    top_1 = float(sorted_probs[0])
    top_5 = float(jnp.sum(sorted_probs[:5]))
    top_10 = float(jnp.sum(sorted_probs[:10]))

    # Effective number of tokens (reciprocal of sum of squared probs)
    eff_tokens = float(1.0 / jnp.sum(probs ** 2).clip(1e-10))

    return {
        "position": position,
        "top_1_prob": top_1,
        "top_5_prob": top_5,
        "top_10_prob": top_10,
        "effective_tokens": eff_tokens,
        "is_concentrated": top_1 > 0.5,
    }


def prediction_landscape_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Combined prediction landscape analysis."""
    profile = logit_distribution_profile(model, tokens, position)
    margin = decision_margin_analysis(model, tokens, position)
    conc = logit_concentration(model, tokens, position)
    return {
        "position": profile["position"],
        "entropy": profile["entropy"],
        "margin_1_2": margin["margin_1_2"],
        "is_confident": margin["is_confident"],
        "top_1_prob": conc["top_1_prob"],
        "effective_tokens": conc["effective_tokens"],
        "logit_std": profile["std"],
    }
