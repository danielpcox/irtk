"""Interpretability benchmarks and quantitative metrics.

Standard metrics for evaluating mechanistic interpretability experiments:
- logit_diff: Difference between correct and incorrect logits
- kl_divergence: KL(original || modified) distribution divergence
- loss_recovered: Fraction of original loss preserved after intervention
- ablation_effect_size: Normalized impact of an ablation
- faithfulness_correlation: How well does an attribution method predict ablation effects
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def logit_diff(
    logits: jnp.ndarray,
    correct_token: int,
    incorrect_token: int,
    pos: int = -1,
) -> float:
    """Compute logit difference between correct and incorrect tokens.

    A standard metric for indirect object identification and similar
    tasks. Positive means the model prefers the correct token.

    Args:
        logits: [..., seq_len, d_vocab] model output logits.
        correct_token: Token ID of the correct answer.
        incorrect_token: Token ID of the incorrect/counterfactual answer.
        pos: Sequence position to evaluate (default: last).

    Returns:
        logit_diff = logits[pos, correct] - logits[pos, incorrect].
    """
    return float(logits[pos, correct_token] - logits[pos, incorrect_token])


def kl_divergence(
    logits_original: jnp.ndarray,
    logits_modified: jnp.ndarray,
    pos: int = -1,
) -> float:
    """Compute KL divergence between original and modified output distributions.

    KL(P || Q) where P = softmax(original), Q = softmax(modified).
    Measures how much information is lost by the modification.

    Args:
        logits_original: Original model logits [..., seq_len, d_vocab].
        logits_modified: Modified model logits (same shape).
        pos: Sequence position to evaluate.

    Returns:
        KL divergence (nonnegative float, 0 = identical distributions).
    """
    log_p = jax.nn.log_softmax(logits_original[pos])
    log_q = jax.nn.log_softmax(logits_modified[pos])
    p = jnp.exp(log_p)
    kl = float(jnp.sum(p * (log_p - log_q)))
    return max(kl, 0.0)  # numerical floor


def loss_recovered(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    modified_logits: jnp.ndarray,
    corrupted_logits: Optional[jnp.ndarray] = None,
    pos: int = -1,
) -> float:
    """Fraction of original loss preserved after intervention.

    loss_recovered = 1 - (L_modified - L_clean) / (L_corrupted - L_clean)

    Values close to 1.0 mean the intervention barely changes behavior.
    Values close to 0.0 mean the intervention is as bad as full corruption.

    Args:
        model: HookedTransformer (used for clean forward pass).
        tokens: Input tokens.
        modified_logits: Logits from the intervention.
        corrupted_logits: Logits from full corruption (if None, uses
            zero-ablation of all attention outputs as a proxy).
        pos: Position to evaluate loss at.

    Returns:
        Fraction of loss recovered (can be > 1 or < 0 in edge cases).
    """
    tokens = jnp.array(tokens)
    clean_logits = model(tokens)

    target = int(tokens[pos]) if pos != -1 else int(tokens[-1])
    # We evaluate the next-token prediction if pos < len-1
    if pos < len(tokens) - 1 and pos >= 0:
        target = int(tokens[pos + 1])

    clean_loss = -float(jax.nn.log_softmax(clean_logits[pos])[target])
    mod_loss = -float(jax.nn.log_softmax(modified_logits[pos])[target])

    if corrupted_logits is not None:
        corrupt_loss = -float(jax.nn.log_softmax(corrupted_logits[pos])[target])
    else:
        # Use uniform distribution as baseline corruption
        corrupt_loss = float(jnp.log(jnp.array(clean_logits.shape[-1], dtype=jnp.float32)))

    denom = corrupt_loss - clean_loss
    if abs(denom) < 1e-10:
        return 1.0

    return float(1.0 - (mod_loss - clean_loss) / denom)


def ablation_effect_size(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    metric_fn: Callable[[jnp.ndarray], float],
    ablation_type: str = "zero",
) -> dict:
    """Compute normalized effect size of ablating a hook point.

    Measures how much ablating this hook changes the metric, normalized
    by the scale of the metric itself.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        hook_name: Hook point to ablate.
        metric_fn: Function(logits) -> float.
        ablation_type: "zero" (replace with zeros) or "mean" (replace with mean).

    Returns:
        Dict with:
        - "clean_metric": baseline metric
        - "ablated_metric": metric after ablation
        - "effect": ablated - clean
        - "effect_size": |effect| / max(|clean|, eps)
    """
    tokens = jnp.array(tokens)
    clean_metric = metric_fn(model(tokens))

    _, cache = model.run_with_cache(tokens)

    if hook_name not in cache.cache_dict:
        return {
            "clean_metric": float(clean_metric),
            "ablated_metric": float(clean_metric),
            "effect": 0.0,
            "effect_size": 0.0,
        }

    cached_act = cache.cache_dict[hook_name]

    if ablation_type == "mean":
        replacement = jnp.mean(cached_act, axis=0, keepdims=True) * jnp.ones_like(cached_act)
        def hook_fn(x, name):
            return replacement
    else:
        def hook_fn(x, name):
            return jnp.zeros_like(x)

    ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    ablated_metric = metric_fn(ablated_logits)

    effect = float(ablated_metric - clean_metric)
    effect_size = abs(effect) / max(abs(float(clean_metric)), 1e-10)

    return {
        "clean_metric": float(clean_metric),
        "ablated_metric": float(ablated_metric),
        "effect": effect,
        "effect_size": effect_size,
    }


def faithfulness_correlation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    attributions: dict[str, float],
    metric_fn: Callable[[jnp.ndarray], float],
) -> dict:
    """Measure how well attribution scores predict actual ablation effects.

    A faithful attribution method should assign high scores to hooks
    whose ablation causes large metric changes. Computes the Pearson
    correlation between attribution scores and actual ablation effects.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        attributions: Dict mapping hook_name -> attribution score.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
        - "correlation": Pearson r between attributions and ablation effects
        - "attribution_scores": list of attribution values
        - "ablation_effects": list of actual effects
        - "hook_names": list of hook names
    """
    tokens = jnp.array(tokens)

    hook_names = list(attributions.keys())
    attr_scores = [attributions[h] for h in hook_names]
    ablation_effects = []

    for hook_name in hook_names:
        result = ablation_effect_size(model, tokens, hook_name, metric_fn)
        ablation_effects.append(abs(result["effect"]))

    attr_arr = np.array(attr_scores, dtype=np.float64)
    abl_arr = np.array(ablation_effects, dtype=np.float64)

    # Pearson correlation
    if len(attr_arr) < 2 or np.std(attr_arr) < 1e-10 or np.std(abl_arr) < 1e-10:
        corr = 0.0
    else:
        corr = float(np.corrcoef(attr_arr, abl_arr)[0, 1])

    return {
        "correlation": corr,
        "attribution_scores": attr_scores,
        "ablation_effects": ablation_effects,
        "hook_names": hook_names,
    }
