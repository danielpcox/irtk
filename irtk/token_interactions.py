"""Pairwise token interaction analysis.

Tools for understanding how input tokens interact to produce model predictions:
- token_interaction_matrix: How much each output position depends on each input token
- pairwise_synergy: Whether two tokens provide redundant or synergistic information
- conditional_attribution: How one token's importance changes given another
- bigram_attention_scores: Which token pairs co-attend across heads
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def token_interaction_matrix(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
) -> np.ndarray:
    """Compute pairwise token interaction matrix via leave-one-out.

    Entry [i] measures how much ablating input token i changes the metric.
    This is the first-order attribution. For second-order interactions,
    use pairwise_synergy.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.

    Returns:
        [seq_len] array of per-token attribution scores (absolute change).
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    baseline = metric_fn(model(tokens))

    attributions = np.zeros(seq_len)
    for pos in range(seq_len):
        def ablate_hook(x, name, p=pos):
            return x.at[p].set(jnp.zeros(x.shape[-1]))

        logits = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", ablate_hook)]
        )
        attributions[pos] = abs(baseline - metric_fn(logits))

    return attributions


def pairwise_synergy(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    positions: Optional[list[int]] = None,
) -> dict:
    """Measure synergy/redundancy between pairs of input tokens.

    For positions i, j, synergy is defined as:
        synergy(i,j) = effect(ablate both) - effect(ablate i) - effect(ablate j)

    Positive synergy means the tokens are synergistic (together they matter
    more than the sum of their individual effects).
    Negative synergy means redundancy (ablating both has less effect than
    the sum of individual ablations).

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        positions: Subset of positions to analyze (default: all).

    Returns:
        Dict with:
        - "synergy_matrix": [n, n] synergy scores between position pairs
        - "individual_effects": [n] per-position ablation effects
        - "positions": the positions analyzed
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    if positions is None:
        positions = list(range(seq_len))

    n = len(positions)
    baseline = metric_fn(model(tokens))

    # Individual effects
    individual = np.zeros(n)
    for idx, pos in enumerate(positions):
        def ablate_hook(x, name, p=pos):
            return x.at[p].set(jnp.zeros(x.shape[-1]))

        logits = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", ablate_hook)]
        )
        individual[idx] = baseline - metric_fn(logits)

    # Pairwise effects
    synergy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = positions[i], positions[j]

            def ablate_both(x, name, _pi=pi, _pj=pj):
                x = x.at[_pi].set(jnp.zeros(x.shape[-1]))
                x = x.at[_pj].set(jnp.zeros(x.shape[-1]))
                return x

            logits = model.run_with_hooks(
                tokens, fwd_hooks=[("hook_embed", ablate_both)]
            )
            joint_effect = baseline - metric_fn(logits)

            # Synergy = joint - sum of individual
            s = joint_effect - individual[i] - individual[j]
            synergy[i, j] = s
            synergy[j, i] = s

    return {
        "synergy_matrix": synergy,
        "individual_effects": individual,
        "positions": positions,
    }


def conditional_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    target_pos: int,
    context_pos: int,
) -> dict:
    """How does target token's importance change when context token is ablated?

    Computes four quantities:
    - effect_target: ablating target alone
    - effect_context: ablating context alone
    - effect_both: ablating both
    - conditional_effect: effect of target given context is ablated

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        target_pos: Position of the token whose importance we measure.
        context_pos: Position of the conditioning token.

    Returns:
        Dict with effect_target, effect_context, effect_both,
        conditional_effect, and importance_change.
    """
    tokens = jnp.array(tokens)
    baseline = metric_fn(model(tokens))

    # Ablate target only
    def ablate_target(x, name):
        return x.at[target_pos].set(jnp.zeros(x.shape[-1]))

    logits_t = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", ablate_target)]
    )
    effect_target = baseline - metric_fn(logits_t)

    # Ablate context only
    def ablate_context(x, name):
        return x.at[context_pos].set(jnp.zeros(x.shape[-1]))

    logits_c = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", ablate_context)]
    )
    metric_without_context = metric_fn(logits_c)
    effect_context = baseline - metric_without_context

    # Ablate both
    def ablate_both(x, name):
        x = x.at[target_pos].set(jnp.zeros(x.shape[-1]))
        x = x.at[context_pos].set(jnp.zeros(x.shape[-1]))
        return x

    logits_both = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", ablate_both)]
    )
    effect_both = baseline - metric_fn(logits_both)

    # Conditional: effect of ablating target when context is already ablated
    conditional_effect = effect_both - effect_context

    return {
        "effect_target": float(effect_target),
        "effect_context": float(effect_context),
        "effect_both": float(effect_both),
        "conditional_effect": float(conditional_effect),
        "importance_change": float(conditional_effect - effect_target),
    }


def bigram_attention_scores(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: Optional[int] = None,
) -> np.ndarray:
    """Compute attention-weighted bigram scores across heads.

    For each pair of positions (q, k), averages the attention weight
    across all heads (and optionally all layers). High scores indicate
    positions that are strongly linked by attention.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: If specified, only use this layer. Otherwise average all layers.

    Returns:
        [seq_len, seq_len] matrix of mean attention scores.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    seq_len = len(tokens)
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    if layer is not None:
        layers = [layer]
    else:
        layers = list(range(n_layers))

    total = np.zeros((seq_len, seq_len))
    count = 0

    for l in layers:
        hook_name = f"blocks.{l}.attn.hook_pattern"
        if hook_name not in cache.cache_dict:
            continue
        # pattern shape: [seq_q, n_heads, seq_k]
        pattern = np.array(cache.cache_dict[hook_name])
        for h in range(n_heads):
            total += pattern[:, h, :]
            count += 1

    if count > 0:
        total /= count

    return total


def token_pair_effect(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    pos_a: int,
    pos_b: int,
) -> dict:
    """Detailed analysis of how two tokens jointly affect the metric.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        pos_a: First token position.
        pos_b: Second token position.

    Returns:
        Dict with clean metric, ablation effects, synergy, and redundancy ratio.
    """
    tokens = jnp.array(tokens)
    clean = metric_fn(model(tokens))

    # Ablate a
    def abl_a(x, name):
        return x.at[pos_a].set(jnp.zeros(x.shape[-1]))
    metric_no_a = metric_fn(model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", abl_a)]))

    # Ablate b
    def abl_b(x, name):
        return x.at[pos_b].set(jnp.zeros(x.shape[-1]))
    metric_no_b = metric_fn(model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", abl_b)]))

    # Ablate both
    def abl_both(x, name):
        x = x.at[pos_a].set(jnp.zeros(x.shape[-1]))
        x = x.at[pos_b].set(jnp.zeros(x.shape[-1]))
        return x
    metric_no_both = metric_fn(model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", abl_both)]))

    effect_a = clean - metric_no_a
    effect_b = clean - metric_no_b
    effect_both = clean - metric_no_both
    synergy = effect_both - effect_a - effect_b

    # Redundancy ratio: if close to 1, effects are additive
    sum_individual = abs(effect_a) + abs(effect_b)
    if sum_individual > 1e-10:
        redundancy_ratio = abs(effect_both) / sum_individual
    else:
        redundancy_ratio = 1.0

    return {
        "clean_metric": float(clean),
        "effect_a": float(effect_a),
        "effect_b": float(effect_b),
        "effect_both": float(effect_both),
        "synergy": float(synergy),
        "redundancy_ratio": float(redundancy_ratio),
    }
