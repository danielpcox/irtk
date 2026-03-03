"""Position-centric attribution for transformer analysis.

Track how information from specific input positions influences
model outputs and intermediate representations.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def position_gradient_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Compute gradient-based attribution for each input position.

    Measures how much each token position contributes to the metric
    by computing the gradient of the metric w.r.t. embeddings.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "position_scores": [seq_len] attribution per position
        - "most_important_position": position with highest attribution
        - "attribution_entropy": entropy of the attribution distribution
        - "concentration_ratio": fraction of total in top position
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    # Get embeddings for gradient computation
    _, cache = model.run_with_cache(tokens)
    if "hook_embed" not in cache.cache_dict:
        return {"position_scores": np.zeros(seq_len), "most_important_position": 0,
                "attribution_entropy": 0.0, "concentration_ratio": 0.0}

    embed = np.array(cache.cache_dict["hook_embed"])  # [seq, d_model]

    # Position-level attribution via embedding norms and metric sensitivity
    scores = np.zeros(seq_len)

    for pos in range(seq_len):
        # Ablate position
        def make_pos_ablate(p):
            def hook(x, name):
                return x.at[p].set(0.0)
            return hook

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", make_pos_ablate(pos))]
        )
        original_logits = model(tokens)
        original = float(metric_fn(original_logits))
        ablated = float(metric_fn(ablated_logits))
        scores[pos] = abs(original - ablated)

    # Normalize
    total = scores.sum()
    if total > 1e-10:
        normed = scores / total
        entropy = float(-np.sum(normed * np.log(normed + 1e-10)))
        concentration = float(np.max(normed))
    else:
        entropy = 0.0
        concentration = 0.0

    return {
        "position_scores": scores,
        "most_important_position": int(np.argmax(scores)),
        "attribution_entropy": entropy,
        "concentration_ratio": concentration,
    }


def position_flow_through_layers(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    source_pos: int,
) -> dict:
    """Trace how a source position's information flows through layers.

    At each layer, measures how much attention each position pays
    to the source, creating a flow map.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        source_pos: Position to trace.

    Returns:
        Dict with:
        - "attention_flow": [n_layers, seq_len] mean attention to source per position per layer
        - "flow_persistence": [n_layers] total attention to source per layer
        - "decay_rate": how quickly attention to source diminishes
        - "max_receiver": position that attends most to source (excluding self)
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    _, cache = model.run_with_cache(tokens)

    flow = np.zeros((n_layers, seq_len))

    for layer in range(n_layers):
        hook = f"blocks.{layer}.attn.hook_pattern"
        if hook not in cache.cache_dict:
            continue
        pat = np.array(cache.cache_dict[hook])

        if pat.ndim == 3:
            # Average across heads: [n_heads, seq, seq] -> [seq, seq]
            mean_pat = np.mean(pat, axis=0)
        else:
            mean_pat = pat

        if source_pos < mean_pat.shape[1]:
            flow[layer] = mean_pat[:seq_len, source_pos]

    persistence = np.sum(flow, axis=1)

    # Decay rate
    if n_layers >= 2 and persistence[0] > 1e-10:
        decay = float(1.0 - persistence[-1] / persistence[0])
    else:
        decay = 0.0

    # Max receiver (excluding self)
    total_flow = np.sum(flow, axis=0)
    total_flow[source_pos] = 0  # exclude self
    max_receiver = int(np.argmax(total_flow))

    return {
        "attention_flow": flow,
        "flow_persistence": persistence,
        "decay_rate": float(decay),
        "max_receiver": max_receiver,
    }


def position_interaction_matrix(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Compute pairwise interaction strengths between positions.

    For each pair (i, j), measures the effect of jointly ablating
    both positions vs. ablating each individually.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "interaction_matrix": [seq_len, seq_len] pairwise interaction
        - "strongest_interaction": (pos_i, pos_j, score)
        - "mean_interaction": average pairwise interaction
        - "individual_effects": [seq_len] single-position ablation effects
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    original_logits = model(tokens)
    original = float(metric_fn(original_logits))

    # Single-position effects
    single = np.zeros(seq_len)
    for pos in range(seq_len):
        def make_ablate(p):
            def hook(x, name):
                return x.at[p].set(0.0)
            return hook
        abl = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", make_ablate(pos))]
        )
        single[pos] = abs(original - float(metric_fn(abl)))

    # Pairwise interactions (superlinear effects)
    interactions = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            def make_pair_ablate(p1, p2):
                def hook(x, name):
                    x = x.at[p1].set(0.0)
                    x = x.at[p2].set(0.0)
                    return x
                return hook

            pair_abl = model.run_with_hooks(
                tokens, fwd_hooks=[("hook_embed", make_pair_ablate(i, j))]
            )
            pair_effect = abs(original - float(metric_fn(pair_abl)))
            # Interaction = joint effect beyond sum of individual effects
            interaction = pair_effect - single[i] - single[j]
            interactions[i, j] = interaction
            interactions[j, i] = interaction

    # Find strongest
    best_i, best_j = 0, 1
    best_score = 0.0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if abs(interactions[i, j]) > abs(best_score):
                best_i, best_j = i, j
                best_score = float(interactions[i, j])

    return {
        "interaction_matrix": interactions,
        "strongest_interaction": (best_i, best_j, best_score),
        "mean_interaction": float(np.mean(np.abs(interactions))),
        "individual_effects": single,
    }


def position_specific_ablation(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    positions: Optional[list] = None,
) -> dict:
    """Measure the impact of ablating each position's contribution.

    For each position, zeros out the embedding and measures metric change
    across layers.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function from logits -> float.
        positions: Positions to test (default: all).

    Returns:
        Dict with:
        - "ablation_effects": [n_positions] metric change per position
        - "most_critical_position": position with largest effect
        - "least_critical_position": position with smallest effect
        - "effect_variance": variance of effects
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)
    test_positions = positions if positions is not None else list(range(seq_len))

    original = float(metric_fn(model(tokens)))
    effects = np.zeros(len(test_positions))

    for idx, pos in enumerate(test_positions):
        def make_ablate(p):
            def hook(x, name):
                return x.at[p].set(0.0)
            return hook

        ablated_logits = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", make_ablate(pos))]
        )
        effects[idx] = abs(original - float(metric_fn(ablated_logits)))

    most = test_positions[int(np.argmax(effects))]
    least = test_positions[int(np.argmin(effects))]

    return {
        "ablation_effects": effects,
        "most_critical_position": most,
        "least_critical_position": least,
        "effect_variance": float(np.var(effects)),
    }


def position_to_logit_attribution(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
) -> dict:
    """Attribute target token prediction to input positions.

    Uses embedding-level ablation to measure how much each input
    position contributes to the target token's logit.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        target_token: Token whose logit we attribute.

    Returns:
        Dict with:
        - "position_attributions": [seq_len] attribution per position
        - "most_contributing": position contributing most to target logit
        - "total_contribution": sum of all position contributions
        - "normalized_attributions": [seq_len] sum-to-one attributions
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    original_logits = np.array(model(tokens))
    original_target = float(original_logits[-1, target_token])

    attrs = np.zeros(seq_len)

    for pos in range(seq_len):
        def make_ablate(p):
            def hook(x, name):
                return x.at[p].set(0.0)
            return hook

        ablated_logits = np.array(model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", make_ablate(pos))]
        ))
        ablated_target = float(ablated_logits[-1, target_token])
        attrs[pos] = original_target - ablated_target

    total = float(np.sum(np.abs(attrs)))
    if total > 1e-10:
        normed = np.abs(attrs) / total
    else:
        normed = np.zeros(seq_len)

    return {
        "position_attributions": attrs,
        "most_contributing": int(np.argmax(np.abs(attrs))),
        "total_contribution": total,
        "normalized_attributions": normed,
    }
