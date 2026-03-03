"""Sequence dynamics analysis.

Analyzes how transformers process sequence structure: repetition handling,
long-range dependencies, position bias, length effects, and boundary
effects at sequence edges.

Functions:
- repetition_handling_analysis: Decompose repetition detection vs memorization
- long_range_dependency_tracking: Track information propagation from distant positions
- position_bias_strength: Quantify recency and distance bias in attention
- length_effect_on_circuits: How circuit behavior changes with sequence length
- boundary_effect_analysis: Characterize start/end-of-sequence edge effects

References:
    - Olsson et al. (2022) "In-context Learning and Induction Heads"
    - Xiao et al. (2023) "Efficient Streaming LMs with Attention Sinks"
    - Press et al. (2022) "Train Short, Test Long"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def repetition_handling_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Analyze how the model handles repeated token patterns.

    Compares attention patterns and predictions for first vs repeated
    occurrences of tokens to distinguish memorization from induction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens (should contain repetitions).

    Returns:
        Dict with:
            "first_occurrence_entropy": mean prediction entropy at first occurrences
            "repeated_occurrence_entropy": mean prediction entropy at repeated occurrences
            "entropy_reduction": how much entropy drops for repeated tokens
            "induction_score": how much attention goes to prior occurrences
            "n_repeated_tokens": number of tokens that appear more than once
    """
    _, cache = model.run_with_cache(tokens)
    logits = np.array(model(tokens))
    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    # Find first and repeated occurrences
    seen = {}
    first_positions = []
    repeat_positions = []

    for i in range(seq_len):
        tok = int(tokens_np[i])
        if tok not in seen:
            seen[tok] = i
            first_positions.append(i)
        else:
            repeat_positions.append(i)

    # Entropy at each position
    def position_entropy(pos):
        l = logits[pos]
        probs = np.exp(l - np.max(l))
        probs = probs / np.sum(probs)
        return -float(np.sum(probs * np.log(probs + 1e-10)))

    first_ent = [position_entropy(p) for p in first_positions if p < seq_len]
    repeat_ent = [position_entropy(p) for p in repeat_positions if p < seq_len]

    first_mean = float(np.mean(first_ent)) if first_ent else 0.0
    repeat_mean = float(np.mean(repeat_ent)) if repeat_ent else 0.0

    # Induction score: attention from repeated positions to prior occurrence
    induction_score = 0.0
    n_counted = 0
    for l in range(model.cfg.n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key in cache.cache_dict:
            pattern = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
            for pos in repeat_positions:
                tok = int(tokens_np[pos])
                prior = seen[tok]
                # Attention from pos to position after prior occurrence
                target = min(prior + 1, pos)
                if target < pos:
                    for h in range(pattern.shape[0]):
                        induction_score += float(pattern[h, pos, target])
                        n_counted += 1

    if n_counted > 0:
        induction_score /= n_counted

    return {
        "first_occurrence_entropy": first_mean,
        "repeated_occurrence_entropy": repeat_mean,
        "entropy_reduction": first_mean - repeat_mean,
        "induction_score": induction_score,
        "n_repeated_tokens": len(repeat_positions),
    }


def long_range_dependency_tracking(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    source_pos: int = 0,
    target_pos: int = -1,
) -> dict:
    """Track information propagation from a source to target position.

    Measures how much attention and residual stream information flows
    from a distant source position to the target.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        source_pos: Source position to track.
        target_pos: Target position (-1 = last).

    Returns:
        Dict with:
            "direct_attention_per_layer": [n_layers] direct attention from target to source
            "indirect_attention_estimate": attention flow estimate via intermediate positions
            "source_ablation_effect": metric change when source position is zeroed
            "information_retention": cosine similarity of source info at target position
            "effective_distance": "effective" number of hops for information transfer
    """
    if target_pos == -1:
        target_pos = len(tokens) - 1

    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Direct attention
    direct_attn = np.zeros(n_layers)
    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key in cache.cache_dict:
            pattern = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
            # Mean attention from target to source across heads
            direct_attn[l] = float(np.mean(pattern[:, target_pos, source_pos]))

    # Indirect attention estimate (sum of 2-hop paths through intermediate positions)
    indirect = 0.0
    if n_layers >= 2:
        for l in range(n_layers - 1):
            k1 = f"blocks.{l}.attn.hook_pattern"
            k2 = f"blocks.{l + 1}.attn.hook_pattern"
            if k1 in cache.cache_dict and k2 in cache.cache_dict:
                p1 = np.array(cache.cache_dict[k1])
                p2 = np.array(cache.cache_dict[k2])
                # Attention: src -> intermediate (layer l) then intermediate -> target (layer l+1)
                for mid in range(len(tokens)):
                    if mid != source_pos and mid != target_pos:
                        path = float(np.mean(p1[:, mid, source_pos]) * np.mean(p2[:, target_pos, mid]))
                        indirect += path

    # Source ablation effect
    hook_name = f"blocks.0.hook_resid_pre"

    def ablate_source(x, name, _pos=source_pos):
        return x.at[_pos, :].set(0.0)

    clean_logits = np.array(model(tokens))
    ablated_logits = np.array(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate_source)]))
    ablation_effect = float(np.linalg.norm(clean_logits[target_pos] - ablated_logits[target_pos]))

    # Information retention: cosine sim between source representation and target residual
    src_key = f"blocks.0.hook_resid_post"
    tgt_key = f"blocks.{n_layers - 1}.hook_resid_post"
    retention = 0.0
    if src_key in cache.cache_dict and tgt_key in cache.cache_dict:
        src_rep = np.array(cache.cache_dict[src_key][source_pos])
        tgt_rep = np.array(cache.cache_dict[tgt_key][target_pos])
        retention = float(np.dot(src_rep, tgt_rep) / (np.linalg.norm(src_rep) * np.linalg.norm(tgt_rep) + 1e-10))

    # Effective distance: reciprocal of total attention flow
    total_flow = float(np.sum(direct_attn)) + indirect
    eff_dist = 1.0 / (total_flow + 1e-10)

    return {
        "direct_attention_per_layer": direct_attn,
        "indirect_attention_estimate": indirect,
        "source_ablation_effect": ablation_effect,
        "information_retention": retention,
        "effective_distance": min(eff_dist, float(target_pos - source_pos)),
    }


def position_bias_strength(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Quantify recency bias and distance-based attention degradation.

    Measures how attention decays with distance and whether the model
    has systematic position biases.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "mean_attention_by_distance": [seq_len-1] mean attention weight at each distance
            "recency_bias": fraction of attention on the most recent 25% of positions
            "primacy_bias": fraction of attention on the first 25% of positions
            "distance_decay_rate": exponential decay constant of attention with distance
            "position_entropy": mean entropy of per-position attention distributions
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    # Aggregate attention patterns
    attention_by_distance = np.zeros(seq_len)
    count_by_distance = np.zeros(seq_len)

    all_patterns = []
    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        if key in cache.cache_dict:
            pattern = np.array(cache.cache_dict[key])  # [n_heads, seq, seq]
            all_patterns.append(pattern)

            for h in range(pattern.shape[0]):
                for q in range(seq_len):
                    for k in range(q + 1):  # Causal: can only attend to k <= q
                        dist = q - k
                        attention_by_distance[dist] += pattern[h, q, k]
                        count_by_distance[dist] += 1

    mean_by_dist = np.zeros(seq_len)
    for d in range(seq_len):
        if count_by_distance[d] > 0:
            mean_by_dist[d] = attention_by_distance[d] / count_by_distance[d]

    # Recency bias: attention on most recent 25%
    quarter = max(1, seq_len // 4)
    recency = float(np.sum(mean_by_dist[:quarter]) / (np.sum(mean_by_dist) + 1e-10))

    # Primacy: fraction of attention to first position (distance from query)
    # This is tricky — primacy is about attending to the FIRST positions
    primacy_attn = 0.0
    total_attn = 0.0
    for pat in all_patterns:
        for h in range(pat.shape[0]):
            for q in range(seq_len):
                # Attention to first 25% of positions
                first_quarter = max(1, (q + 1) // 4)
                primacy_attn += float(np.sum(pat[h, q, :first_quarter]))
                total_attn += float(np.sum(pat[h, q, :q + 1]))
    primacy = primacy_attn / (total_attn + 1e-10)

    # Distance decay rate
    valid = mean_by_dist[:max(2, seq_len // 2)]
    valid = valid[valid > 1e-10]
    if len(valid) >= 2:
        log_attn = np.log(valid + 1e-10)
        decay_rate = -float((log_attn[-1] - log_attn[0]) / (len(valid) - 1))
    else:
        decay_rate = 0.0

    # Position entropy
    entropies = []
    for pat in all_patterns:
        for h in range(pat.shape[0]):
            for q in range(1, seq_len):
                row = pat[h, q, :q + 1]
                row = row / (np.sum(row) + 1e-10)
                ent = -float(np.sum(row * np.log(row + 1e-10)))
                entropies.append(ent)
    pos_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "mean_attention_by_distance": mean_by_dist,
        "recency_bias": recency,
        "primacy_bias": primacy,
        "distance_decay_rate": decay_rate,
        "position_entropy": pos_entropy,
    }


def length_effect_on_circuits(
    model: HookedTransformer,
    base_tokens: jnp.ndarray,
    metric_fn: Callable,
    lengths: Optional[list] = None,
) -> dict:
    """Test how circuit behavior changes with sequence length.

    Evaluates the metric at different context lengths to identify
    length-dependent behavior changes.

    Args:
        model: HookedTransformer.
        base_tokens: [max_seq_len] input tokens.
        metric_fn: Function(logits) -> float.
        lengths: Lengths to test. Defaults to powers of 2 up to len(tokens).

    Returns:
        Dict with:
            "lengths": list of tested lengths
            "metrics": [n_lengths] metric at each length
            "metric_variance": variance of metric across lengths
            "length_sensitivity": normalized metric change per unit length
            "stable_beyond": length beyond which metric stabilizes
    """
    max_len = len(base_tokens)
    if lengths is None:
        lengths = [2 ** i for i in range(1, 20) if 2 ** i <= max_len]
        if max_len not in lengths:
            lengths.append(max_len)

    metrics = []
    for l in lengths:
        truncated = base_tokens[:l]
        logits = model(truncated)
        metrics.append(float(metric_fn(logits)))

    metrics = np.array(metrics)

    # Length sensitivity
    if len(metrics) >= 2:
        diffs = np.abs(np.diff(metrics))
        length_diffs = np.diff(lengths)
        sensitivity = float(np.mean(diffs / (np.array(length_diffs, dtype=float) + 1e-10)))
    else:
        sensitivity = 0.0

    # Stable beyond: first length where subsequent metrics don't change much
    stable = lengths[-1]
    if len(metrics) >= 2:
        for i in range(len(metrics) - 1):
            if all(abs(metrics[j] - metrics[i]) < 0.01 * abs(metrics[i] + 1e-10)
                   for j in range(i + 1, len(metrics))):
                stable = lengths[i]
                break

    return {
        "lengths": lengths,
        "metrics": metrics,
        "metric_variance": float(np.var(metrics)),
        "length_sensitivity": sensitivity,
        "stable_beyond": stable,
    }


def boundary_effect_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict:
    """Characterize start/end-of-sequence edge effects.

    Analyzes how residual stream norms, attention patterns, and
    prediction confidence differ at sequence boundaries vs middle.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.

    Returns:
        Dict with:
            "start_norm_ratio": ratio of residual norm at position 0 vs mean
            "end_norm_ratio": ratio of residual norm at last position vs mean
            "start_attention_concentration": mean attention received by position 0
            "end_confidence_boost": confidence boost at final position vs mean
            "boundary_effect_strength": overall boundary effect measure
    """
    _, cache = model.run_with_cache(tokens)
    logits = np.array(model(tokens))
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    # Residual norms at last layer
    key = f"blocks.{n_layers - 1}.hook_resid_post"
    if key in cache.cache_dict:
        resid = np.array(cache.cache_dict[key])
        norms = np.linalg.norm(resid, axis=-1)
        mean_norm = float(np.mean(norms))
        start_ratio = float(norms[0] / (mean_norm + 1e-10))
        end_ratio = float(norms[-1] / (mean_norm + 1e-10))
    else:
        start_ratio = 1.0
        end_ratio = 1.0

    # Attention received by position 0
    start_attn = 0.0
    n_patterns = 0
    for l in range(n_layers):
        pkey = f"blocks.{l}.attn.hook_pattern"
        if pkey in cache.cache_dict:
            pattern = np.array(cache.cache_dict[pkey])
            for h in range(pattern.shape[0]):
                start_attn += float(np.mean(pattern[h, :, 0]))
                n_patterns += 1
    start_attn = start_attn / (n_patterns + 1e-10)

    # Prediction confidence
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    confidences = np.max(probs, axis=-1)
    mean_conf = float(np.mean(confidences))
    end_boost = float(confidences[-1] / (mean_conf + 1e-10))

    # Overall boundary effect
    boundary_strength = abs(start_ratio - 1.0) + abs(end_ratio - 1.0) + abs(end_boost - 1.0)

    return {
        "start_norm_ratio": start_ratio,
        "end_norm_ratio": end_ratio,
        "start_attention_concentration": start_attn,
        "end_confidence_boost": end_boost,
        "boundary_effect_strength": boundary_strength,
    }
