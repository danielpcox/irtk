"""Context sensitivity analysis.

Analyze how model behavior depends on token position, context length,
local vs. distant information, and in-context learning dynamics.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def positional_attention_profile(
    model: HookedTransformer,
    token_sequences: list,
) -> dict:
    """Compute attention weight as a function of relative token distance.

    For each head, measures how much attention weight is placed on tokens
    at each relative position (1 back, 2 back, etc.).

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.

    Returns:
        Dict with:
        - "profiles": [n_layers, n_heads, max_distance] mean attention by distance
        - "max_distance": maximum distance analyzed
        - "dominant_distances": [n_layers, n_heads] distance with most attention
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Determine max sequence length
    max_seq = max(len(jnp.array(t)) for t in token_sequences)
    max_dist = max_seq

    # Accumulate attention by distance
    profiles = np.zeros((n_layers, n_heads, max_dist))
    counts = np.zeros((n_layers, n_heads, max_dist))

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        seq_len = len(tokens)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            hook = f"blocks.{layer}.attn.hook_pattern"
            if hook not in cache.cache_dict:
                continue
            pattern = np.array(cache.cache_dict[hook])  # [n_heads, seq, seq]
            if pattern.ndim == 2:
                pattern = pattern[np.newaxis]

            for h in range(min(n_heads, pattern.shape[0])):
                for i in range(seq_len):
                    for j in range(i + 1):  # causal: j <= i
                        dist = i - j
                        if dist < max_dist:
                            profiles[layer, h, dist] += pattern[h, i, j]
                            counts[layer, h, dist] += 1

    # Normalize
    valid = counts > 0
    profiles[valid] /= counts[valid]

    # Dominant distances
    dominant = np.argmax(profiles, axis=2)

    return {
        "profiles": profiles,
        "max_distance": max_dist,
        "dominant_distances": dominant,
    }


def local_vs_global_score(
    model: HookedTransformer,
    token_sequences: list,
    local_window: int = 5,
) -> dict:
    """Quantify local vs. global attention for each head.

    Measures what fraction of each head's attention falls within
    a local window vs. distant context.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        local_window: Size of local window (positions back).

    Returns:
        Dict with:
        - "local_fractions": [n_layers, n_heads] fraction of attention in local window
        - "global_fractions": [n_layers, n_heads] fraction outside local window
        - "most_local_heads": list of (layer, head) sorted by local fraction
        - "most_global_heads": list of (layer, head) sorted by global fraction
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    local_sum = np.zeros((n_layers, n_heads))
    total_sum = np.zeros((n_layers, n_heads))

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        seq_len = len(tokens)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            hook = f"blocks.{layer}.attn.hook_pattern"
            if hook not in cache.cache_dict:
                continue
            pattern = np.array(cache.cache_dict[hook])
            if pattern.ndim == 2:
                pattern = pattern[np.newaxis]

            for h in range(min(n_heads, pattern.shape[0])):
                for i in range(seq_len):
                    for j in range(i + 1):
                        w = pattern[h, i, j]
                        total_sum[layer, h] += w
                        if i - j <= local_window:
                            local_sum[layer, h] += w

    local_frac = local_sum / np.maximum(total_sum, 1e-10)
    global_frac = 1.0 - local_frac

    # Rank heads
    flat_local = local_frac.flatten()
    local_order = np.argsort(flat_local)[::-1]
    most_local = [(int(i // n_heads), int(i % n_heads)) for i in local_order]

    flat_global = global_frac.flatten()
    global_order = np.argsort(flat_global)[::-1]
    most_global = [(int(i // n_heads), int(i % n_heads)) for i in global_order]

    return {
        "local_fractions": local_frac,
        "global_fractions": global_frac,
        "most_local_heads": most_local,
        "most_global_heads": most_global,
    }


def context_length_sensitivity(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    min_prefix: int = 1,
) -> dict:
    """Measure how predictions change as context length grows.

    Runs the model with increasing prefix lengths and tracks the metric.

    Args:
        model: HookedTransformer.
        tokens: Full token sequence.
        metric_fn: Function from logits -> float.
        min_prefix: Minimum prefix length to start from.

    Returns:
        Dict with:
        - "lengths": [n] context lengths tested
        - "metrics": [n] metric value at each length
        - "convergence_length": length where metric stabilizes (within 5% of final)
        - "max_change_length": length with largest metric change from previous
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    lengths = []
    metrics = []

    for l in range(max(min_prefix, 1), seq_len + 1):
        prefix = tokens[:l]
        logits = model(prefix)
        m = float(metric_fn(logits))
        lengths.append(l)
        metrics.append(m)

    metrics_arr = np.array(metrics)
    final_metric = metrics_arr[-1]

    # Convergence: first length within 5% of final
    convergence = lengths[-1]
    for i, m in enumerate(metrics_arr):
        if abs(final_metric) > 1e-10:
            if abs(m - final_metric) / abs(final_metric) < 0.05:
                convergence = lengths[i]
                break
        else:
            if abs(m - final_metric) < 0.05:
                convergence = lengths[i]
                break

    # Max change
    if len(metrics_arr) > 1:
        changes = np.abs(np.diff(metrics_arr))
        max_change_idx = int(np.argmax(changes))
        max_change_length = lengths[max_change_idx + 1]
    else:
        max_change_length = lengths[0]

    return {
        "lengths": np.array(lengths),
        "metrics": metrics_arr,
        "convergence_length": convergence,
        "max_change_length": max_change_length,
    }


def in_context_learning_dynamics(
    model: HookedTransformer,
    example_tokens_list: list,
    query_tokens: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Track how representations shift as in-context examples are added.

    Measures how the query's representation changes at a specific hook
    as more ICL examples are prepended.

    Args:
        model: HookedTransformer.
        example_tokens_list: List of token arrays, each an ICL example.
        query_tokens: Query token array.
        hook_name: Hook to monitor.
        pos: Position in query to track (-1 for last).

    Returns:
        Dict with:
        - "n_examples": [n+1] number of examples (0 = query only)
        - "representations": [n+1, d_model] activation at each step
        - "cosine_shifts": [n] cosine distance from previous step
        - "cumulative_shift": [n+1] cosine distance from 0-example baseline
        - "most_impactful_example": which example caused largest shift
    """
    query_tokens = jnp.array(query_tokens)
    d_model = model.cfg.d_model

    representations = []
    n_steps = len(example_tokens_list) + 1

    # Build incrementally longer contexts
    for n_ex in range(n_steps):
        if n_ex == 0:
            combined = query_tokens
        else:
            parts = []
            for i in range(n_ex):
                parts.append(jnp.array(example_tokens_list[i]))
            parts.append(query_tokens)
            combined = jnp.concatenate(parts)

        _, cache = model.run_with_cache(combined)
        if hook_name in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook_name])
            resolved = pos if pos >= 0 else acts.shape[0] + pos
            if 0 <= resolved < acts.shape[0]:
                representations.append(acts[resolved])
            else:
                representations.append(np.zeros(d_model))
        else:
            representations.append(np.zeros(d_model))

    reps = np.array(representations)

    # Cosine shifts between consecutive steps
    cosine_shifts = []
    for i in range(1, len(reps)):
        norm_a = np.linalg.norm(reps[i - 1])
        norm_b = np.linalg.norm(reps[i])
        if norm_a > 1e-10 and norm_b > 1e-10:
            cos = float(np.dot(reps[i - 1], reps[i]) / (norm_a * norm_b))
            cosine_shifts.append(1.0 - cos)  # distance
        else:
            cosine_shifts.append(0.0)

    # Cumulative shift from baseline
    cumulative = []
    baseline = reps[0]
    baseline_norm = np.linalg.norm(baseline)
    for i in range(len(reps)):
        norm_i = np.linalg.norm(reps[i])
        if baseline_norm > 1e-10 and norm_i > 1e-10:
            cos = float(np.dot(baseline, reps[i]) / (baseline_norm * norm_i))
            cumulative.append(1.0 - cos)
        else:
            cumulative.append(0.0)

    most_impactful = int(np.argmax(cosine_shifts)) if cosine_shifts else 0

    return {
        "n_examples": np.arange(n_steps),
        "representations": reps,
        "cosine_shifts": np.array(cosine_shifts),
        "cumulative_shift": np.array(cumulative),
        "most_impactful_example": most_impactful,
    }


def token_distance_effect(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_pos: int,
    metric_fn: Callable,
) -> dict:
    """Measure how ablating tokens at different distances affects the metric.

    Produces a distance-vs-importance curve revealing the effective
    receptive field at a given position.

    Args:
        model: HookedTransformer.
        tokens: Token sequence.
        target_pos: Position whose prediction we're analyzing.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "distances": [n] distances from target position
        - "effects": [n] metric change when ablating token at each distance
        - "effective_window": distance containing 90% of total effect
        - "peak_distance": distance with largest effect
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)
    resolved_pos = target_pos if target_pos >= 0 else seq_len + target_pos

    # Baseline
    _, cache = model.run_with_cache(tokens)
    embed_hook = "hook_embed"
    if embed_hook not in cache.cache_dict:
        return {"distances": np.array([]), "effects": np.array([]),
                "effective_window": 0, "peak_distance": 0}

    clean_embed = np.array(cache.cache_dict[embed_hook])
    full_logits = model(tokens)
    full_metric = float(metric_fn(full_logits))

    distances = []
    effects = []

    for pos in range(resolved_pos + 1):  # Only causal positions
        dist = resolved_pos - pos
        distances.append(dist)

        # Ablate this position's embedding
        ablated = jnp.array(clean_embed.copy())
        ablated = ablated.at[pos].set(0.0)

        def make_hook(emb):
            def hook(x, name):
                return emb
            return hook

        logits = model.run_with_hooks(tokens, fwd_hooks=[(embed_hook, make_hook(ablated))])
        ablated_metric = float(metric_fn(logits))
        effects.append(full_metric - ablated_metric)

    distances = np.array(distances)
    effects = np.array(effects)

    # Sort by distance
    order = np.argsort(distances)
    distances = distances[order]
    effects = effects[order]

    # Effective window: distance containing 90% of total effect
    total_effect = np.sum(np.abs(effects))
    cumsum = np.cumsum(np.abs(effects))
    effective_window = len(distances) - 1
    for i in range(len(cumsum)):
        if total_effect > 1e-10 and cumsum[i] >= 0.9 * total_effect:
            effective_window = int(distances[i])
            break

    peak_distance = int(distances[np.argmax(np.abs(effects))]) if len(effects) > 0 else 0

    return {
        "distances": distances,
        "effects": effects,
        "effective_window": effective_window,
        "peak_distance": peak_distance,
    }
